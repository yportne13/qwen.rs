use candle_core::{Tensor, Error};

use self::{model::Qwen, quantized::ModelWeights};

pub mod tiktoken;
pub mod model;
pub mod quantized;

pub trait QwenModel {
    fn apply(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor, Error>;
}

impl QwenModel for Qwen {
    fn apply(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor, Error> {
        self.forward(x, index_pos)
    }
}

impl QwenModel for ModelWeights {
    fn apply(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor, Error> {
        self.forward(x, index_pos)
    }
}
