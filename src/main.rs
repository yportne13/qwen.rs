use std::{str::FromStr, io::Write};

use anyhow::Error;
use candle_core::{Tensor, quantized::ggml_file};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use qwen::QwenModel;

use crate::qwen::{tiktoken::qwen_base, model::{Qwen, Config}, quantized::ModelWeights};

mod qwen;

fn readline_loop(mut f: impl FnMut(String) -> Result<(), Error>) -> Result<(), Error> {
    let stdin = std::io::stdin();
    let mut buf = String::new();

    loop {
        let _ = stdin.read_line(&mut buf)?;
        if buf == "quit\r\n" {
            break;
        } else {
            f(format!(r#"<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"#, buf))?;
        }
    }

    Ok(())
}

fn chat<T: QwenModel>(qwen: &mut T) -> Result<(), Error> {

    let system_prompt = r#"<|im_start|>system
You are a helpful assistant.<|im_end|>
"#;

    let bpe = qwen_base().unwrap();
    let tokens = bpe.encode_with_special_tokens(
      system_prompt
    ).into_iter().map(|x| x as u32).collect::<Vec<_>>();

    let device = candle_core::Device::Cpu;
    let mut logits_processor = LogitsProcessor::new(0, Some(0.0), Some(0.5));

    //////////////////
    let mut index_pos = 0;
    let input = Tensor::new(&tokens[..], &device)?.unsqueeze(0)?;
    let _ = qwen.apply(&input, index_pos)?;
    index_pos += tokens.len();
    print!(">>> ");
    std::io::stdout().flush()?;
    readline_loop(|s| {
        let mut token = bpe.encode_with_special_tokens(
            &s
        ).into_iter().map(|x| x as u32).collect::<Vec<_>>();
        loop {
            let input = Tensor::new(&token[..], &device)?.unsqueeze(0)?;
            let logits = qwen.apply(&input, index_pos)?;
            let logits = logits.squeeze(0)?;
            index_pos += token.len();

            let next_token = logits_processor.sample(&logits)?;
            token = vec![next_token];

            if let Ok(text) = bpe.decode(vec![next_token as usize]) {
                let text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "");
                print!("{text}");
                std::io::stdout().flush()?;
            }
            if next_token == 151643 {//<|endoftext|>
                break;
            }
        }
        print!(">>> ");
        std::io::stdout().flush()?;
        Ok(())
    })?;
    
    Ok(())
}

fn main() -> Result<(), Error> {
    println!("想跑非量化版还是量化版？输入 0 选择非量化版，输入 2 选择 ggml 量化版");
    let stdin = std::io::stdin();
    let mut buf = String::new();
    let _ = stdin.read_line(&mut buf)?;
    if buf.strip_prefix('0').is_some() {
        println!("准备运行非量化版本");
        println!("如果你还没有权重文件，前往 huggingface 或 modelscope 下载那 8 个后缀是 \".safetensors\"");
        println!("huggingface: https://huggingface.co/Qwen/Qwen-7B-Chat/tree/main");
        println!("modelscope: https://modelscope.cn/models/qwen/Qwen-7B-Chat/files");
        println!("在当前目录下创建名叫 weight 的文件夹，然后把这八个文件放进去即可");
        println!("准备加载模型");
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
        let config = Config::config_7b_v1_1(false, false);
        let vb = VarBuilder::from_safetensors(tensors, candle_core::DType::F16, &device);
        let cache = qwen::model::Cache::new(true, candle_core::DType::F16, &config, &device)?;
        let mut qwen = Qwen::load(vb, &cache, &config)?;
        chat(&mut qwen)?;
    } else if buf.strip_prefix('1').is_some() {
        //println!("准备运行量化版本");
        println!("这是一个尚未完成的 auto-gptq 量化版本");
        println!("如果你还没有权重文件，前往 huggingface 或 modelscope 下载那 3 个后缀是 \".safetensors\"");
        println!("huggingface: https://huggingface.co/Qwen/Qwen-7B-Chat/tree/main");
        println!("modelscope: https://modelscope.cn/models/qwen/Qwen-7B-Chat/files");
        println!("在当前目录下创建名叫 weight 的文件夹，然后把这八个文件放进去即可");
        println!("准备加载模型");
        let filenames = vec![
            std::path::PathBuf::from_str("weight/model-00001-of-00003.safetensors").unwrap(),
            std::path::PathBuf::from_str("weight/model-00002-of-00003.safetensors").unwrap(),
            std::path::PathBuf::from_str("weight/model-00003-of-00003.safetensors").unwrap(),
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
        let config = Config::config_7b_v1_1(false, true);
        let vb = VarBuilder::from_safetensors(tensors, candle_core::DType::F16, &device);
        let cache = qwen::model::Cache::new(true, candle_core::DType::F16, &config, &device)?;
        let mut qwen = Qwen::load(vb, &cache, &config)?;
        chat(&mut qwen)?;
    } else {
        println!("准备运行 ggml 量化版本");
        println!("选择量化位宽");
        println!("1: F16，请确认你有如下文件：weight/qwen7b-ggml-f16.bin");
        println!("2: Q4_0，请确认你有如下文件：weight/qwen7b-ggml-q4_0.bin");
        println!("6: Q5_0，请确认你有如下文件：weight/qwen7b-ggml-q5_0.bin");
        println!("8: Q8_0，请确认你有如下文件：weight/qwen7b-ggml-q8_0.bin");
        let stdin = std::io::stdin();
        let mut buf = String::new();
        let _ = stdin.read_line(&mut buf)?;
        let mut qwen = {
            let mut file = if buf.strip_prefix('1').is_some() {
                std::fs::File::open("weight/qwen7b-ggml-f16.bin")?
            } else if buf.strip_prefix('2').is_some() {
                std::fs::File::open("weight/qwen7b-ggml-q4_0.bin")?
            } else if buf.strip_prefix('6').is_some() {
                std::fs::File::open("weight/qwen7b-ggml-q5_0.bin")?
            } else {
                std::fs::File::open("weight/qwen7b-ggml-q8_0.bin")?
            };
            let model = ggml_file::Content::read(&mut file)?;
            println!("params: {:?}", model.hparams);
            ModelWeights::from_ggml(model, 1)?
        };
        chat(&mut qwen)?;
    }

    Ok(())
}
