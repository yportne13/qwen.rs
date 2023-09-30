use std::collections::HashMap;

use candle_core::quantized::QTensor;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module};

pub const MAX_SEQ_LEN: usize = 8192;

struct RmsNorm {
    inner: candle_nn::LayerNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn new(scale: QTensor, eps: f32) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let scale = scale.dequantize(&Device::Cpu)?;
        let inner = candle_nn::LayerNorm::rms_norm(scale, eps as f64);
        Ok(Self { inner, span })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

// QMatMul wrapper adding some tracing.
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Self {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor);
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Self { inner, span }
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

struct LayerWeights {
    c_attn: QMatMul,
    c_attn_bias: Tensor,
    c_attn_proj: QMatMul,
    ln_1: RmsNorm,
    c_w1: QMatMul,
    c_w2: QMatMul,
    c_proj: QMatMul,
    ln_2: RmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    cos: Tensor,
    sin: Tensor,
    kv_cache: Option<(Tensor, Tensor)>,
    span_attn: tracing::Span,
    span_rot: tracing::Span,
    span_mlp: tracing::Span,
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

impl LayerWeights {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, n_head, seq_len, n_embd) = x.dims4()?;
        let cos = self
            .cos
            .narrow(0, index_pos, seq_len)?
            .reshape((seq_len, n_embd / 2, 1))?;
        let sin = self
            .sin
            .narrow(0, index_pos, seq_len)?
            .reshape((seq_len, n_embd / 2, 1))?;
        let cos = cos.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
        let sin = sin.broadcast_as((b_sz, 1, seq_len, n_embd / 2, 1))?;
        // This mimics the llama.cpp behavior.
        // https://github.com/ggerganov/llama.cpp/blob/1f0bccb27929e261744c979bc75114955da49e98/ggml.c#L12104-L12105
        // The x0 and x1 value are interleaved on the n_embd (= head_dim) dimension.
        // The resulting y0 and y1 are also interleaved with:
        //   y0 = x0*cos - x1*sin
        //   y1 = x0*sin + x1*cos
        let x = x.reshape((b_sz, n_head, seq_len, n_embd / 2, 2))?;
        let x0 = x.narrow(D::Minus1, 0, 1)?;
        let x1 = x.narrow(D::Minus1, 1, 1)?;
        let y0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
        let y1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;
        let rope = Tensor::cat(&[y0, y1], D::Minus1)?;
        let rope = rope.flatten_from(D::Minus2)?;
        Ok(rope)
    }

    fn forward_attn(&mut self, x: &Tensor, mask: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, n_embd) = x.dims3()?;
        let mixed_x_layer = (self.c_attn.forward(x)?).add(&self.c_attn_bias)?;
        let qkv = mixed_x_layer.chunk(3, 2)?;

        let q = qkv[0]
            .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
            .transpose(1, 2)?;
        let k = qkv[1]
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;
        let v = qkv[2]
            .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let k = self.apply_rotary_emb(&k, index_pos)?;

        let (k, v) = match &self.kv_cache {
            None => (k, v),
            Some((k_cache, v_cache)) => {
                if index_pos == 0 {
                    (k, v)
                } else {
                    let k = Tensor::cat(&[k_cache, &k], 2)?.contiguous()?;
                    let v = Tensor::cat(&[v_cache, &v], 2)?.contiguous()?;
                    (k, v)
                }
            }
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Support for MQA, useful for 70B models.
        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
        let mask = mask.broadcast_as(att.shape())?;
        let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
        let att = candle_nn::ops::softmax_last_dim(&att)?;
        // Convert to contiguous as matmul doesn't support strided vs for now.
        let y = att.matmul(&v.contiguous()?)?;
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, n_embd])?;
        let y = self.c_attn_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.n_head / self.n_kv_head;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }
}

pub struct ModelWeights {
    wte: Embedding,
    layers: Vec<LayerWeights>,
    ln_f: RmsNorm,
    lm_head: QMatMul,
    masks: HashMap<usize, Tensor>,
    span: tracing::Span,
    span_output: tracing::Span,
}

fn precomput_freqs_cis(head_dim: usize, freq_base: f32) -> Result<(Tensor, Tensor)> {
    let theta: Vec<_> = (0..head_dim)
        .step_by(2)
        .map(|i| 1f32 / freq_base.powf(i as f32 / head_dim as f32))
        .collect();
    let theta = Tensor::new(theta.as_slice(), &Device::Cpu)?;
    let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, &Device::Cpu)?
        .to_dtype(DType::F32)?
        .reshape((MAX_SEQ_LEN, 1))?
        .matmul(&theta.reshape((1, theta.elem_count()))?)?;
    let cos = idx_theta.cos()?;
    let sin = idx_theta.sin()?;
    Ok((cos, sin))
}

impl ModelWeights {
    #[allow(unused)]
    pub fn from_ggml(mut ct: ggml_file::Content, gqa: usize) -> Result<Self> {
        let cpu = &Device::Cpu;
        let head_dim = (ct.hparams.n_embd / ct.hparams.n_head) as usize;
        let (cos, sin) = precomput_freqs_cis(head_dim, 10000.)?;
        let wte = ct.remove("transformer.wte.weight")?;
        let wte = wte.dequantize(cpu)?;
        let mut layers = Vec::with_capacity(ct.hparams.n_layer as usize);
        for layer_idx in 0..ct.hparams.n_layer {
            let prefix = format!("transformer.h.{layer_idx}");
            let c_attn = ct.remove(&format!("{prefix}.attn.c_attn.weight"))?;
            let c_attn_bias = ct.remove(&format!("{prefix}.attn.c_attn.bias"))?;
            let c_attn_bias = c_attn_bias.dequantize(cpu)?;
            let c_attn_proj = ct.remove(&format!("{prefix}.attn.c_proj.weight"))?;
            let c_w1 = ct.remove(&format!("{prefix}.mlp.w1.weight"))?;
            let c_w2 = ct.remove(&format!("{prefix}.mlp.w2.weight"))?;
            let c_proj = ct.remove(&format!("{prefix}.mlp.c_proj.weight"))?;
            let ln_1 = ct.remove(&format!("{prefix}.ln_1.weight"))?;
            let ln_2 = ct.remove(&format!("{prefix}.ln_2.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                c_attn: QMatMul::from_qtensor(c_attn),
                c_attn_bias,
                c_attn_proj: QMatMul::from_qtensor(c_attn_proj),
                ln_1: RmsNorm::new(ln_1, 1e-5)?,
                c_w1: QMatMul::from_qtensor(c_w1),
                c_w2: QMatMul::from_qtensor(c_w2),
                c_proj: QMatMul::from_qtensor(c_proj),
                ln_2: RmsNorm::new(ln_2, 1e-5)?,
                n_head: ct.hparams.n_head as usize,
                n_kv_head: ct.hparams.n_head as usize / gqa,
                head_dim: (ct.hparams.n_embd / ct.hparams.n_head) as usize,
                cos: cos.clone(),
                sin: sin.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let ln_f = RmsNorm::new(ct.remove("transformer.ln_f.weight")?, 1e-5)?;
        let lm_head = ct.remove("lm_head.weight")?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            wte: Embedding::new(wte, ct.hparams.n_embd as usize),
            layers,
            ln_f,
            lm_head: QMatMul::from_qtensor(lm_head),
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    #[allow(unused)]
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
    ) -> Result<Self> {
        let cpu = &Device::Cpu;
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        // Parameter extraction from metadata.
        let head_count = md_get("llama.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("llama.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("llama.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("llama.embedding_length")?.to_u32()? as usize;
        let rope_dim = md_get("llama.rope.dimension_count")?.to_u32()? as usize;
        // Strangely this value is generally 1e-6 in GGUF file but used to be 1e-5 by default.
        let rms_norm_eps = md_get("llama.attention.layer_norm_rms_epsilon")?.to_f32()?;

        let rope_freq_base = md_get("llama.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(10000f32);
        let (cos, sin) = precomput_freqs_cis(rope_dim, rope_freq_base)?;

        let wte = ct.tensor(reader, "transformer.wte.weight")?;
        let wte = wte.dequantize(cpu)?;
        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("transformer.h.{layer_idx}");
            let c_attn = ct.tensor(reader, &format!("{prefix}.attn.c_attn.weight"))?;
            let c_attn_bias = ct.tensor(reader, &format!("{prefix}.attn.c_attn.bias"))?;
            let c_attn_bias = c_attn_bias.dequantize(cpu)?;
            let c_attn_proj = ct.tensor(reader, &format!("{prefix}.attn.c_proj.weight"))?;
            let c_w1 = ct.tensor(reader, &format!("{prefix}.mlp.w1.weight"))?;
            let c_w2 = ct.tensor(reader, &format!("{prefix}.mlp.w2.weight"))?;
            let c_proj = ct.tensor(reader, &format!("{prefix}.mlp.c_proj.weight"))?;
            let ln_1 = ct.tensor(reader, &format!("{prefix}.ln_1.weight"))?;
            let ln_2 = ct.tensor(reader, &format!("{prefix}.ln_2.weight"))?;
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");
            layers.push(LayerWeights {
                c_attn: QMatMul::from_qtensor(c_attn),
                c_attn_bias,
                c_attn_proj: QMatMul::from_qtensor(c_attn_proj),
                ln_1: RmsNorm::new(ln_1, rms_norm_eps)?,
                c_w1: QMatMul::from_qtensor(c_w1),
                c_w2: QMatMul::from_qtensor(c_w2),
                c_proj: QMatMul::from_qtensor(c_proj),
                ln_2: RmsNorm::new(ln_2, rms_norm_eps)?,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: embedding_length / head_count,
                cos: cos.clone(),
                sin: sin.clone(),
                kv_cache: None,
                span_attn,
                span_rot,
                span_mlp,
            })
        }
        let ln_f = RmsNorm::new(ct.tensor(reader, "transformer.ln_f.weight")?, rms_norm_eps)?;
        let lm_head = ct.tensor(reader, "lm_head.weight")?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            wte: Embedding::new(wte, embedding_length),
            layers,
            ln_f,
            lm_head: QMatMul::from_qtensor(lm_head),
            masks: HashMap::new(),
            span,
            span_output,
        })
    }

    fn mask(&mut self, t: usize) -> Result<Tensor> {
        if let Some(mask) = self.masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &Device::Cpu)?;
            self.masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mask = self.mask(seq_len)?;
        let _enter = self.span.enter();
        let mut layer_in = self.wte.forward(x)?;
        for layer in self.layers.iter_mut() {
            let x = layer_in;
            let residual = &x;
            let x = layer.ln_1.forward(&x)?;
            let attn = layer.forward_attn(&x, &mask, index_pos)?;
            let x = (attn + residual)?;

            // MLP
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ln_2.forward(&x)?;
            let w1 = layer.c_w1.forward(&x)?;
            let w2 = layer.c_w2.forward(&x)?;
            let mlp = layer
                .c_proj
                .forward(&(w1 * candle_nn::ops::silu(&w2)?)?)?;
            layer_in = (mlp + residual)?;
        }
        let x = self.ln_f.forward(&layer_in)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let _enter = self.span_output.enter();
        self.lm_head.forward(&x)
    }
}
