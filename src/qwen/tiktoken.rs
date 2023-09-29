use std::collections::HashMap;

use anyhow::Error;
use base64::{engine::general_purpose, Engine};
use tiktoken_rs::CoreBPE;

pub fn qwen_base() -> Result<CoreBPE, Error> {
    let bpe_file = include_str!("../../assets/qwen.tiktoken");

    let mut encoder = HashMap::default();
    for line in bpe_file.lines() {
        let mut parts = line.split(' ');
        let raw = parts.next().unwrap();
        let token = &general_purpose::STANDARD.decode(raw)?;
        let rank: usize = parts.next().unwrap().parse().unwrap();
        encoder.insert(token.clone(), rank);
    }

    let mut special_tokens = HashMap::default();
    special_tokens.insert(String::from("<|endoftext|>"), 151643);
    special_tokens.insert(String::from("<|im_start|>"), 151644);
    special_tokens.insert(String::from("<|im_end|>"), 151645);
    for i in 0..205 {
        special_tokens.insert(format!("<|extra_{}|>", i), 151646 + i);
    }

    let bpe = CoreBPE::new(
        encoder,
        special_tokens,
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+",
    )?;
    Ok(bpe)
}
