// Copyright (C) 2025 Noah Cashin <noahc959@icloud.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.

// * Unified model abstraction for SmolLM (Transformer) and RWKV7 backends.
// * Provides a common trait `LanguageModelSession` used by the compressor and scan logic.

use anyhow::{bail, Context, Result};
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{
    Cache as LlamaCache, Config as LlamaRuntimeConfig, Llama as LlamaModel, LlamaConfig,
};

// Note: BackendKind removed from public API; keep sessions targeted via concrete types.

pub trait LanguageModelSession {
    fn vocab_size(&self) -> usize;
    fn max_context_length(&self) -> usize;
    fn step_logits_tensor(&mut self, token_id: u32) -> Result<Tensor>;
    fn reprime_with_history_and_get_last_logits_tensor(&mut self, history: &[u32]) -> Result<Tensor>;
}

// ---------------- SmolLM (LLaMA-like) ----------------

pub struct SmolLmSession {
    device: Device,
    model: LlamaModel,
    cache: LlamaCache,
    runtime_cfg: LlamaRuntimeConfig,
    dtype: DType,
    vocab: usize,
    index_pos: usize,
}

impl SmolLmSession {
    pub fn load(weights_paths: &[std::path::PathBuf], config_path: &std::path::Path, device: Device, vocab_size: usize) -> Result<Self> {
        let cfg_bytes = std::fs::read(config_path)
            .with_context(|| format!("failed reading {}", config_path.display()))?;
        let llama_cfg: LlamaConfig = serde_json::from_slice(&cfg_bytes)
            .with_context(|| "failed to parse config.json as HF LlamaConfig")?;
        let runtime_cfg: LlamaRuntimeConfig = llama_cfg.into_config(false);
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(weights_paths, dtype, &device)? };
        let model = LlamaModel::load(vb.clone(), &runtime_cfg)
            .with_context(|| "failed constructing LLaMA model from safetensors")?;
        let cache = LlamaCache::new(true, dtype, &runtime_cfg, &device)
            .with_context(|| "failed to create KV cache")?;
        Ok(Self { device, model, cache, runtime_cfg, dtype, vocab: vocab_size, index_pos: 0 })
    }
}

impl SmolLmSession {
    pub fn index_pos(&self) -> usize { self.index_pos }

    pub fn bounds_for_symbol_on_device(&self, logits: &Tensor, sym: usize) -> Result<(f64, f64)> {
        if sym >= self.vocab {
            bail!("symbol {} out of bounds for vocab size {}", sym, self.vocab);
        }
        let dims = logits.dims();
        if dims.len() != 1 {
            bail!("expected 1D logits tensor, got {:?}", dims);
        }
        let tensor_len = dims[0];
        if tensor_len != self.vocab {
            bail!("logits tensor size {} doesn't match vocab size {}", tensor_len, self.vocab);
        }

        let logits_vec = logits.to_vec1::<f32>()?;
        let pdf = crate::softmax_pdf_floor(&logits_vec, self.vocab, crate::ac_p_min());
        let p_sym = pdf[sym];
        let c_lo = if sym == 0 { 0.0 } else { pdf[0..sym].iter().sum::<f64>() };
        let c_hi = c_lo + p_sym;
        Ok((c_lo, c_hi))
    }
}

impl LanguageModelSession for SmolLmSession {
    fn vocab_size(&self) -> usize { self.vocab }
    fn max_context_length(&self) -> usize { 512 }
    fn step_logits_tensor(&mut self, token_id: u32) -> Result<Tensor> {
        let x = Tensor::new(&[token_id], &self.device)?.reshape((1, 1))?;
        let logits = self.model.forward(&x, self.index_pos, &mut self.cache)?;
        self.index_pos += 1;
        let t = match logits.rank() {
            1 => logits,
            2 => logits.i((logits.dim(0)? - 1, ..))?,
            3 => logits.i((0, logits.dim(1)? - 1, ..))?,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
    }
    fn reprime_with_history_and_get_last_logits_tensor(&mut self, history: &[u32]) -> Result<Tensor> {
        self.cache = LlamaCache::new(true, self.dtype, &self.runtime_cfg, &self.device)
            .with_context(|| "failed to reset KV cache")?;
        self.index_pos = 0;
        if history.is_empty() { bail!("reprime called with empty history"); }
        let x = Tensor::new(history, &self.device)?.reshape((1, history.len()))?;
        let logits = self.model.forward(&x, self.index_pos, &mut self.cache)?;
        self.index_pos += history.len();
        let t = match logits.rank() {
            1 => logits,
            2 => logits.i((logits.dim(0)? - 1, ..))?,
            3 => logits.i((0, logits.dim(1)? - 1, ..))?,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
    }
}

// ---------------- RWKV7 via internal library ----------------

pub struct Rwkv7Session {
    inner: candlerwkv7::models::rwkv7::Model,
    state: candlerwkv7::models::rwkv7::State,
    config: candlerwkv7::models::rwkv7::Config,
    device: Device,
}

impl Rwkv7Session {
    pub fn load(model_path: &std::path::Path, config_path: &std::path::Path, device: Device) -> Result<Self> {
        let cfg_bytes = std::fs::read(config_path)
            .with_context(|| format!("failed reading {}", config_path.display()))?;
        let config: candlerwkv7::models::rwkv7::Config = serde_json::from_slice(&cfg_bytes)
            .with_context(|| "failed to parse config.json as RWKV7 Config")?;
        let tensors = candle_core::safetensors::load(model_path, &device)
            .with_context(|| format!("failed loading safetensors from {}", model_path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = candlerwkv7::models::rwkv7::Model::new(&config, vb)
            .with_context(|| "failed constructing RWKV7 model from safetensors")?;
        let state = candlerwkv7::models::rwkv7::State::new(1, &config, None, &device)
            .with_context(|| "failed to create RWKV7 state")?;
        Ok(Self { inner: model, state, config, device })
    }
}

impl LanguageModelSession for Rwkv7Session {
    fn vocab_size(&self) -> usize { self.config.vocab_size }
    fn max_context_length(&self) -> usize { usize::MAX }
    fn step_logits_tensor(&mut self, token_id: u32) -> Result<Tensor> {
        let x = Tensor::new(&[[token_id]], &self.device)?;
        let logits = self.inner.forward(&x, &mut self.state)?;
        let t = match logits.rank() {
            1 => logits,
            2 => logits.i((logits.dim(0)? - 1, ..))?,
            3 => logits.i((0, logits.dim(1)? - 1, ..))?,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
    }
    fn reprime_with_history_and_get_last_logits_tensor(&mut self, history: &[u32]) -> Result<Tensor> {
        // Reset state and feed sequentially
        self.state = candlerwkv7::models::rwkv7::State::new(1, &self.config, None, &self.device)?;
        if history.is_empty() { bail!("reprime called with empty history"); }
        let mut last = None;
        for &tok in history {
            let x = Tensor::new(&[[tok]], &self.device)?;
            last = Some(self.inner.forward(&x, &mut self.state)?);
        }
        let logits = last.unwrap();
        let t = match logits.rank() {
            1 => logits,
            2 => logits.i((logits.dim(0)? - 1, ..))?,
            3 => logits.i((0, logits.dim(1)? - 1, ..))?,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
    }
}

impl Rwkv7Session {
    #[allow(dead_code)]
    pub fn estimate_state_entropy(&self) -> Result<f32> {
        let mut total = 0.0f32;
        let mut count = 0usize;
        for layer in &self.state.per_layer {
            let mean = layer.att_state.mean_all()?;
            let centered = layer.att_state.broadcast_sub(&mean)?;
            let var = centered.sqr()?.mean_all()?;
            let std = var.sqrt()?.to_scalar::<f32>()?;
            total += std;
            count += 1;
        }
        if count == 0 { return Ok(0.0); }
        Ok(total / (count as f32))
    }
}


