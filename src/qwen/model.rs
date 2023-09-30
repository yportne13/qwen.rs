use candle_core::{DType, Device, IndexOp, Result, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder, Dropout};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

pub const MAX_SEQ_LEN: usize = 8192;

pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub emb_dropout_prob: f32,
    pub attn_dropout_prob: f64,
    pub layer_norm_epsilon: f64,
    pub initializer_range: f64,
    pub max_position_embeddings: usize,
    pub scale_attn_weights: bool,
    pub use_cache: bool,
    pub bf16: bool,
    pub fp16: bool,
    pub fp32: bool,
    pub kv_channels: usize,
    pub rotary_pct: f64,
    pub rotary_emb_base: f32,
    pub use_dynamic_ntk: bool,
    pub use_logn_attn: bool,
    //pub use_flash_attn: String,
    pub intermediate_size: usize,
    pub no_bias: bool,
    pub tie_word_embeddings: bool,
    pub use_cache_quantization: bool,
    pub use_cache_kernel: bool,
}

impl Config {
    pub fn config_7b_v1_1(_use_flash_attn: bool) -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            emb_dropout_prob: 0.0,
            attn_dropout_prob: 0.0,
            layer_norm_epsilon: 1e-6,
            initializer_range: 0.02,
            max_position_embeddings: 8192,
            scale_attn_weights: true,
            use_cache: true,
            bf16: false,
            fp16: false,
            fp32: false,
            kv_channels: 128,
            rotary_pct: 1.0,
            rotary_emb_base: 10000.0,
            use_dynamic_ntk: true,
            use_logn_attn: true,
            //use_flash_attn: "auto",
            intermediate_size: 22016,
            no_bias: true,
            tie_word_embeddings: false,
            use_cache_quantization: false,
            use_cache_kernel: false,
        }
    }
}


#[derive(Clone)]
pub struct Cache {
    masks: Arc<Mutex<HashMap<(usize, usize), Tensor>>>,
    pub use_kv_cache: bool,
    #[allow(clippy::type_complexity)]
    kvs: Arc<Mutex<Vec<Option<(Tensor, Tensor)>>>>,
    cos: Tensor,
    sin: Tensor,
    device: Device,
}

impl Cache {
    pub fn new(use_kv_cache: bool, dtype: DType, config: &Config, device: &Device) -> Result<Self> {
        // precompute freqs_cis
        let n_elem = config.hidden_size / config.num_attention_heads;
        let theta: Vec<_> = (0..n_elem)
            .step_by(2)
            .map(|i| 1f32 / config.rotary_emb_base.powf(i as f32 / n_elem as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0, MAX_SEQ_LEN as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        // This is different from the paper, see:
        // https://github.com/huggingface/transformers/blob/6112b1c6442aaf7affd2b0676a1cd4eee30c45cf/src/transformers/models/llama/modeling_llama.py#L112
        let idx_theta = Tensor::cat(&[&idx_theta, &idx_theta], D::Minus1)?;
        let cos = idx_theta.cos()?.to_dtype(dtype)?;
        let sin = idx_theta.sin()?.to_dtype(dtype)?;
        Ok(Self {
            masks: Arc::new(Mutex::new(HashMap::new())),
            use_kv_cache,
            kvs: Arc::new(Mutex::new(vec![None; config.num_hidden_layers])),
            device: device.clone(),
            cos,
            sin,
        })
    }

    fn mask(&self, t1: usize, t2: usize) -> Result<Tensor> {
        let mut masks = self.masks.lock().unwrap();
        if let Some(mask) = masks.get(&(t1, t2)) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t1)
                .flat_map(|i| (0..t2).map(move |j| u8::from(j > i + (t2 - t1))))
                .collect();
            let mask = Tensor::from_slice(&mask, (t1, t2), &self.device)?;
            masks.insert((t1, t2), mask.clone());
            Ok(mask)
        }
    }
}

// We wrap the `Linear` layer here to add some tracing so that it's easier to profile the resulting
// model.
#[derive(Debug)]
pub struct Linear {
    inner: candle_nn::Linear,
    span: tracing::Span,
}

impl Linear {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

fn linear(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let span = tracing::span!(tracing::Level::TRACE, "linear");
    let inner = candle_nn::linear_no_bias(size1, size2, vb)?;
    Ok(Linear { inner, span })
}

fn linear_bias(size1: usize, size2: usize, vb: VarBuilder) -> Result<Linear> {
    let span = tracing::span!(tracing::Level::TRACE, "linear_bias");
    let inner = candle_nn::linear(size1, size2, vb)?;
    Ok(Linear { inner, span })
}

struct RmsNorm {
    inner: candle_nn::RmsNorm,
    span: tracing::Span,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let inner = candle_nn::rms_norm(size, eps, vb)?;
        Ok(Self { inner, span })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(x)
    }
}

struct CausalSelfAttention {
    c_attn: Linear,
    c_proj: Linear,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,
    cache: Cache,
    use_flash_attn: bool,
    span: tracing::Span,
    span_rot: tracing::Span,
}

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<Tensor> {
    candle_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

impl CausalSelfAttention {
    fn apply_rotary_emb(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let _enter = self.span_rot.enter();
        let (b_sz, _, seq_len, hidden_size) = x.dims4()?;
        let cos = self.cache.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.cache.sin.narrow(0, index_pos, seq_len)?;
        let cos = cos.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let sin = sin.broadcast_as((b_sz, 1, seq_len, hidden_size))?;
        let x1 = x.narrow(D::Minus1, 0, hidden_size / 2)?;
        let x2 = x.narrow(D::Minus1, hidden_size / 2, hidden_size / 2)?;
        let rotate_x = Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)?;
        let rope = (x.broadcast_mul(&cos)? + rotate_x.broadcast_mul(&sin)?)?;
        Ok(rope)
    }

    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b_sz, seq_len, hidden_size) = x.dims3()?;
        let mixed_x_layer = self.c_attn.forward(x)?;
        let qkv = mixed_x_layer.chunk(3, 2)?;

        let q = qkv[0]
            .reshape((b_sz, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = qkv[1]
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;
        let mut v = qkv[2]
            .reshape((b_sz, seq_len, self.num_key_value_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q = self.apply_rotary_emb(&q, index_pos)?;
        let mut k = self.apply_rotary_emb(&k, index_pos)?;

        if self.cache.use_kv_cache {
            let mut cache = self.cache.kvs.lock().unwrap();
            if let Some((cache_k, cache_v)) = &cache[block_idx] {
                k = Tensor::cat(&[cache_k, &k], 2)?.contiguous()?;
                v = Tensor::cat(&[cache_v, &v], 2)?.contiguous()?;
                let k_seq_len = k.dims()[1];
                if k_seq_len > MAX_SEQ_LEN {
                    k = k
                        .narrow(D::Minus1, k_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
                let v_seq_len = v.dims()[1];
                if v_seq_len > 2 * MAX_SEQ_LEN {
                    v = v
                        .narrow(D::Minus1, v_seq_len - MAX_SEQ_LEN, MAX_SEQ_LEN)?
                        .contiguous()?
                }
            }
            cache[block_idx] = Some((k.clone(), v.clone()))
        }

        let k = self.repeat_kv(k)?;
        let v = self.repeat_kv(v)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let q = q.transpose(1, 2)?;
            let k = k.transpose(1, 2)?;
            let v = v.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&q, &k, &v, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = q.dtype();
            let q = q.to_dtype(DType::F32)?;
            let k = k.to_dtype(DType::F32)?;
            let v = v.to_dtype(DType::F32)?;
            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;
            let mask = self.cache.mask(seq_len, *att.shape().dims().get(3).unwrap())?.broadcast_as(att.shape())?;
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            let att = candle_nn::ops::softmax(&att, D::Minus1)?;
            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?.reshape(&[b_sz, seq_len, hidden_size])?;
        let y = self.c_proj.forward(&y)?;
        Ok(y)
    }

    fn repeat_kv(&self, x: Tensor) -> Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_key_value_heads;
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

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "attn");
        let span_rot = tracing::span!(tracing::Level::TRACE, "attn-rot");
        let c_attn = linear_bias(
            cfg.hidden_size,
            3 * cfg.kv_channels * cfg.num_attention_heads,
            vb.pp("c_attn")
        )?;
        let c_proj = linear(
            cfg.hidden_size,
            cfg.kv_channels * cfg.num_attention_heads,
            vb.pp("c_proj")
        )?;//TODO:!cfg.no_bias
        Ok(Self {
            c_attn,
            c_proj,
            num_attention_heads: cfg.num_attention_heads,
            num_key_value_heads: cfg.num_attention_heads,
            head_dim: cfg.hidden_size / cfg.num_attention_heads,
            cache: cache.clone(),
            use_flash_attn: false,//TODO:
            span,
            span_rot,
        })
    }
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

struct QwenMlp {
    c_w1: Linear,
    c_w2: Linear,
    c_proj: Linear,
    span: tracing::Span,
}

impl QwenMlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let x = (self.c_w1.forward(x)? * candle_nn::ops::silu(&self.c_w2.forward(x)?)?)?;
        self.c_proj.forward(&x)
    }

    fn load(vb: VarBuilder, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        let h_size = cfg.hidden_size;
        let i_size = cfg.intermediate_size/2;
        let c_w1 = linear(h_size, i_size, vb.pp("w1"))?;//TODO:!cfg.no_bias
        let c_w2 = linear(h_size, i_size, vb.pp("w2"))?;//TODO:!cfg.no_bias
        let c_proj = linear(i_size, h_size, vb.pp("c_proj"))?;//TODO:!cfg.no_bias
        Ok(Self {
            c_w1,
            c_w2,
            c_proj,
            span,
        })
    }
}

struct QwenBlock {
    rms_1: RmsNorm,
    attn: CausalSelfAttention,
    rms_2: RmsNorm,
    mlp: QwenMlp,
    span: tracing::Span,
}

impl QwenBlock {
    fn forward(&self, x: &Tensor, index_pos: usize, block_idx: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let residual = x;
        let x = self.rms_1.forward(x)?;
        let x = (self.attn.forward(&x, index_pos, block_idx)? + residual)?;
        let residual = &x;
        let x = (self.mlp.forward(&self.rms_2.forward(&x)?)? + residual)?;
        Ok(x)
    }

    fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "block");
        let attn = CausalSelfAttention::load(vb.pp("attn"), cache, cfg)?;
        let mlp = QwenMlp::load(vb.pp("mlp"), cfg)?;
        let rms_1 = RmsNorm::load(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("ln_1"))?;
        let rms_2 = RmsNorm::load(
            cfg.hidden_size,
            cfg.layer_norm_epsilon,
            vb.pp("ln_2"),
        )?;
        Ok(Self {
            rms_1,
            attn,
            rms_2,
            mlp,
            span,
        })
    }
}

pub struct Qwen {
    wte: Embedding,
    dropout: Dropout,
    blocks: Vec<QwenBlock>,
    ln_f: RmsNorm,
    lm_head: Linear,
}

impl Qwen {
    pub fn forward(&self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (_b_sz, seq_len) = x.dims2()?;
        let mut x = self.wte.forward(x)?;
        x = self.dropout.forward(&x, true)?;
        for (block_idx, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x, index_pos, block_idx)?;
        }
        let x = self.ln_f.forward(&x)?;
        let x = x.i((.., seq_len - 1, ..))?;
        let logits = self.lm_head.forward(&x)?;
        logits.to_dtype(DType::F32)
    }

    pub fn load(vb: VarBuilder, cache: &Cache, cfg: &Config) -> Result<Self> {
        let embeddings = vb.pp("transformer.wte").get((cfg.vocab_size, cfg.hidden_size), "weight")?;
        let wte = Embedding::new(embeddings, cfg.hidden_size);
        let dropout = Dropout::new(cfg.emb_dropout_prob);
        let blocks: Vec<_> = (0..cfg.num_hidden_layers)
            .map(|i| QwenBlock::load(vb.pp(&format!("transformer.h.{i}")), cache, cfg).unwrap())//, cache
            .collect();
        let ln_f = RmsNorm::load(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("transformer.ln_f"))?;
        let lm_head = linear(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))?;

        Ok(Self {
            wte,
            dropout,
            blocks,
            ln_f,
            lm_head,
        })
    }
}
