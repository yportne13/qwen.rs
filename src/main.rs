use std::{str::FromStr, io::Write};

use anyhow::Error;
use candle_core::Tensor;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;

use crate::qwen::{tiktoken::qwen_base, model::{Qwen, Config}};


mod qwen;

fn main() -> Result<(), Error> {
    let prompt = "你好";
    let bpe = qwen_base().unwrap();
    let mut tokens = bpe.encode_with_special_tokens(
      prompt
    ).into_iter().map(|x| x as u32).collect::<Vec<_>>();
    //println!("{:?}", tokens);

    let filenames = vec![
        std::path::PathBuf::from_str("weight/model-00001-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00002-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00003-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00004-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00005-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00006-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00007-of-00008.safetensors").unwrap(),
        std::path::PathBuf::from_str("weight/model-00008-of-00008.safetensors").unwrap(),
    ];
    let handles = filenames
        .iter()
        .map(|f| Ok(unsafe { candle_core::safetensors::MmapedFile::new(f.as_path())? }))
        .collect::<Result<Vec<_>, Error>>()?;
    let tensors: Vec<_> = handles
        .iter()
        .map(|h| Ok(h.deserialize()?))
        .collect::<Result<Vec<_>, Error>>()?;

    let device = candle_core::Device::Cpu;
    let config = Config::config_7b_v1_1(false);
    let vb = VarBuilder::from_safetensors(tensors, candle_core::DType::BF16, &device);
    let cache = qwen::model::Cache::new(true, candle_core::DType::BF16, &config, &device)?;
    let qwen = Qwen::load(vb, &cache, &config)?;

    //////////////////
    println!("starting the inference loop");
    print!("{prompt}");
    let mut logits_processor = LogitsProcessor::new(0, Some(0.0), Some(0.5));
    let start_gen = std::time::Instant::now();
    let mut index_pos = 0;
    let mut token_generated = 0;
    for index in 0..100 {
        let context_size = if cache.use_kv_cache && index > 0 {
            1
        } else {
            tokens.len()
        };
        let ctxt = &tokens[tokens.len().saturating_sub(context_size)..];
        let input = Tensor::new(ctxt, &device)?.unsqueeze(0)?;
        let logits = qwen.forward(&input, index_pos)?;
        let logits = logits.squeeze(0)?;
        index_pos += ctxt.len();

        let next_token = logits_processor.sample(&logits)?;
        token_generated += 1;
        tokens.push(next_token);

        // Extracting the last token as a string is complicated, here we just apply some simple
        // heuristics as it seems to work well enough for this example. See the following for more
        // details:
        // https://github.com/huggingface/tokenizers/issues/1141#issuecomment-1562644141
        if let Ok(text) = bpe.decode(vec![next_token as usize]) {
            let text = text.replace('▁', " ").replace("<0x0A>", "\n");
            print!("{text}");
            std::io::stdout().flush()?;
        }
        if next_token == 151643 {
            break;
        }
    }
    let dt = start_gen.elapsed();
    println!(
        "\n\n{} tokens generated ({} token/s)\n",
        token_generated,
        token_generated as f64 / dt.as_secs_f64(),
    );

    Ok(())
}
