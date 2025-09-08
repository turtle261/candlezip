use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::process::{Command, Stdio};
use std::io::{self as stdio, BufRead};
// removed: unused collections imports

use anyhow::{bail, Context, Result};
use blake3::Hasher;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::{arg, Parser, Subcommand, ValueHint};
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokenizers::Tokenizer;
use chrono::Utc;
use csv;

// Candle
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use candle_transformers::models::llama::{
    Cache as LlamaCache, Config as LlamaRuntimeConfig, Llama as LlamaModel, LlamaConfig,
};

// -------------------------------
// Container format (very small header + AC payload)
// -------------------------------
//
// Magic (u32 LE): 0x5a505447  "GPTZ"
// Version (u16 LE): 1
// Header JSON length (u32 LE)
// Header JSON bytes (UTF-8)
//
// The JSON header has:
// {
//   "model_repo": "HuggingFaceTB/SmolLM-135M",
//   "model_file": "model.safetensors",
//   "tokenizer_repo": "HuggingFaceTB/SmolLM-135M",
//   "tokenizer_file": "tokenizer.json",
//   "model_blake3": "<64-hex>",
//   "tokenizer_blake3": "<64-hex>",
//   "coder_base": 256",
//   "coder_precision": 8,
//   "vocab_size": <u32>,
//   "bos_token_id": <u32>,
//   "token_count": <u64>,
//   "orig_len_bytes": <u64>,
//   "orig_blake3": "<64-hex>",
// }

const MAGIC: u32 = 0x5a505447; // "GPTZ"
const VERSION: u16 = 2;

// Arithmetic coder params (base-2, bitstream output)
const AC_BASE: u32 = 2;
const AC_PRECISION: u32 = 32;

// Defaults (HF Hub)
const DEFAULT_MODEL_REPO: &str = "HuggingFaceTB/SmolLM2-135M-instruct";
const DEFAULT_MODEL_FILE: &str = "auto"; // "auto" downloads model.safetensors or resolves shards via index.json
const DEFAULT_TOKENIZER_REPO: &str = "HuggingFaceTB/SmolLM2-135M-instruct";
const DEFAULT_TOKENIZER_FILE: &str = "tokenizer.json";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HeaderJson {}

#[derive(Parser, Debug)]
#[command(name="gptzip", about="Model-based text compressor (Candle + HF Transformers LLaMA + 64-bit arithmetic coder)")]
struct Cli {
    /// Force running on CPU (default tries CUDA 0 then CPU)
    #[arg(long)]
    cpu: bool,

    /// Optional path to local model weights (.safetensors or .index.json). If absent, download from --model-repo/--model-file
    #[arg(long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,

    /// HF repo id for model (if downloading)
    #[arg(long, default_value = DEFAULT_MODEL_REPO)]
    model_repo: String,

    /// File name inside the model repo (if downloading): "model.safetensors", "*.index.json", or "auto"
    #[arg(long, default_value = DEFAULT_MODEL_FILE)]
    model_file: String,

    /// Path to tokenizer.json (local). If absent, download from --tokenizer-repo/--tokenizer-file
    #[arg(long, value_hint = ValueHint::FilePath)]
    tokenizer: Option<PathBuf>,

    /// HF repo id for tokenizer (if downloading).
    #[arg(long, default_value = DEFAULT_TOKENIZER_REPO)]
    tokenizer_repo: String,

    /// File name inside tokenizer repo (if downloading)
    #[arg(long, default_value = DEFAULT_TOKENIZER_FILE)]
    tokenizer_file: String,

    /// Allow decode even if model/tokenizer BLAKE3 do not match the container
    #[arg(long)]
    force_mismatch: bool,

    /// Sliding attention context (lookback+current). Default 512 for reliability and speed.
    #[arg(long, default_value = "512")]
    context: usize,

    /// Reprime interval in tokens to avoid per-token window replays (default equals context)
    #[arg(long, default_value = "512")]
    reprime_interval: usize,

    /// Enable entropy scan with external agent; produces proof.csv and meta.json
    #[arg(long, default_value_t = false)]
    scan: bool,

    /// Scan chunk size in tokens (default: reprime_interval)
    #[arg(long)]
    scan_chunk_size: Option<usize>,

    /// Maximum agent steps per chunk (planning/tool invocations)
    #[arg(long, default_value_t = 7)]
    scan_max_steps: usize,

    /// Max hint tokens used for conditioning when computing cross-entropy
    #[arg(long, default_value_t = 128)]
    scan_max_hint_tokens: usize,

    /// Path to MCP config used by the agent (default: agent/mcp_config.json)
    #[arg(long, value_hint = ValueHint::FilePath, default_value = "agent/mcp_config.json")]
    scan_mcp_config: PathBuf,

    /// Output directory for scan artifacts (proof.csv, meta.json, logs)
    #[arg(long, value_hint = ValueHint::DirPath, default_value = "scan_output")]
    scan_output_dir: PathBuf,

    /// Python executable to invoke agent ("auto" tries python then py)
    #[arg(long, default_value = "auto")]
    scan_python: String,

    /// Stream agent logs verbosely to terminal
    #[arg(long, default_value_t = true)]
    scan_verbose: bool,

    /// Lookahead window in tokens for cross-entropy evaluation per chunk
    #[arg(long, default_value_t = 512)]
    scan_lookahead: usize,

    /// Hard timeout (seconds) for the agent per chunk; kills agent on expiry
    #[arg(long, default_value_t = 120)]
    scan_agent_timeout: u64,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compress a UTF-8 text file into a .gptz container
    Compress {
        #[arg(value_hint = ValueHint::FilePath)]
        input: PathBuf,
        #[arg(value_hint = ValueHint::FilePath)]
        output: PathBuf,
    },

    /// Decompress a .gptz container back to a UTF-8 text file
    Decompress {
        #[arg(value_hint = ValueHint::FilePath)]
        input: PathBuf,
        #[arg(value_hint = ValueHint::FilePath)]
        output: PathBuf,
    },

    /// Quick encode->decode self test (prints bpb)
    SelfTest {
        #[arg(value_hint = ValueHint::FilePath)]
        input: PathBuf,
    },
}

// -------------------------------
// Small utils
// -------------------------------

// removed: hex helpers (use binary 16-byte truncation instead)

fn detect_device(force_cpu: bool) -> Device {
    if force_cpu {
        Device::Cpu
    } else {
        // Try CUDA:0 then fallback to CPU.
        match Device::new_cuda(0) {
            Ok(d) => d,
            Err(_) => Device::Cpu,
        }
    }
}

// -------------------------------
// 64-bit arithmetic coder on base-2 (bitstream output)
// -------------------------------

fn ac_p_min() -> f64 {
    // 2 * base^-(precision-2) (tiny but nonzero)
    let base = AC_BASE as f64;
    2.0 * base.powi(-(AC_PRECISION as i32 - 2))
}

struct ArithmeticEncoder<W: Write> {
    b_to_pm1: u64,
    b_to_pm2: u64,
    mask: u64,
    low: u64,
    high: u64,
    carry_run: u64,
    out: W,
    bit_buffer: u8,
    bit_count: u8,
    bytes_out: u64,
}

impl<W: Write> ArithmeticEncoder<W> {
    fn new(out: W) -> Self {
        let base = AC_BASE as u64;
        let precision = AC_PRECISION as u32;
        let full = (base.pow(precision)) - 1;
        Self {
            b_to_pm1: base.pow(precision - 1),
            b_to_pm2: base.pow(precision - 2),
            mask: full,
            low: 0,
            high: full,
            carry_run: 0,
            out,
            bit_buffer: 0,
            bit_count: 0,
            bytes_out: 0,
        }
    }

    fn write_byte(&mut self, byte: u8) -> Result<()> {
        self.out.write_all(&[byte])?;
        self.bytes_out += 1;
        Ok(())
    }

    fn put_bit_internal(&mut self, bit: u8) -> Result<()> {
        // MSB-first packing
        self.bit_buffer = (self.bit_buffer << 1) | (bit & 1);
        self.bit_count += 1;
        if self.bit_count == 8 {
            let b = self.bit_buffer;
            self.write_byte(b)?;
            self.bit_buffer = 0;
            self.bit_count = 0;
        }
        Ok(())
    }

    fn put_bit(&mut self, bit: u8) -> Result<()> {
        self.put_bit_internal(bit)?;
        while self.carry_run > 0 {
            self.put_bit_internal((!bit) & 1)?;
            self.carry_run -= 1;
        }
        Ok(())
    }

    #[allow(dead_code)]
    fn encode_symbol(&mut self, sym: usize, pdf: &[f64]) -> Result<()> {
        // Build CDF in [0,1]
        let mut c_lo = 0.0f64;
        for i in 0..sym { c_lo += pdf[i]; }
        let mut c_hi = c_lo + pdf[sym];
        if c_hi > 1.0 { c_hi = 1.0; }

        let low_f = self.low as f64;
        let high_f = self.high as f64;
        let range = high_f - low_f + 1.0;
        let new_low = (low_f + (range * c_lo).floor()) as u64;
        let new_high = (low_f + (range * c_hi).floor() - 1.0) as u64;

        self.low = new_low;
        self.high = new_high;

        // Renormalize (emit common MSBs) with base-2 (bit) coder
        loop {
            if self.high < self.b_to_pm1 {
                // MSB 0
                self.put_bit(0)?;
            } else if self.low >= self.b_to_pm1 {
                // MSB 1
                self.put_bit(1)?;
                self.low -= self.b_to_pm1;
                self.high -= self.b_to_pm1;
            } else if self.low >= self.b_to_pm2 && self.high < self.b_to_pm2 * 3 {
                // E3 underflow condition
                self.carry_run += 1;
                self.low -= self.b_to_pm2;
                self.high -= self.b_to_pm2;
            } else {
                break;
            }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
        }
        Ok(())
    }

    fn encode_interval(&mut self, c_lo: f64, c_hi: f64) -> Result<()> {
        let cl = c_lo.max(0.0).min(1.0);
        let ch = c_hi.max(cl).min(1.0);

        let low_f = self.low as f64;
        let high_f = self.high as f64;
        let range = high_f - low_f + 1.0;
        let new_low = (low_f + (range * cl).floor()) as u64;
        let new_high = (low_f + (range * ch).floor() - 1.0) as u64;

        self.low = new_low;
        self.high = new_high;

        // Renormalize (emit common MSBs) with base-2 (bit) coder
        loop {
            if self.high < self.b_to_pm1 {
                self.put_bit(0)?;
            } else if self.low >= self.b_to_pm1 {
                self.put_bit(1)?;
                self.low -= self.b_to_pm1;
                self.high -= self.b_to_pm1;
            } else if self.low >= self.b_to_pm2 && self.high < self.b_to_pm2 * 3 {
                // E3 underflow condition
                self.carry_run += 1;
                self.low -= self.b_to_pm2;
                self.high -= self.b_to_pm2;
            } else {
                break;
            }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        // Termination: emit one more bit to disambiguate current interval
        self.carry_run += 1;
        if self.low < self.b_to_pm2 {
            self.put_bit(0)?;
        } else {
            self.put_bit(1)?;
        }
        // Flush remaining bits in buffer with zeros
        if self.bit_count > 0 {
            let remaining = 8 - self.bit_count;
            for _ in 0..remaining { self.put_bit_internal(0)?; }
        }
        Ok(())
    }

    fn bytes_written(&self) -> u64 { self.bytes_out }
}

struct ArithmeticDecoder<'a> {
    b_to_pm1: u64,
    b_to_pm2: u64,
    mask: u64,
    low: u64,
    high: u64,
    code: u64,
    input: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> ArithmeticDecoder<'a> {
    fn new(input: &'a [u8]) -> Result<Self> {
        let base = AC_BASE as u64;
        let precision = AC_PRECISION as u32;
        let mut s = Self {
            b_to_pm1: base.pow(precision - 1),
            b_to_pm2: base.pow(precision - 2),
            mask: base.pow(precision) - 1,
            low: 0,
            high: base.pow(precision) - 1,
            code: 0,
            input,
            byte_pos: 0,
            bit_pos: 0,
        };
        for _ in 0..precision {
            s.code = (s.code << 1) | (s.get_bit().unwrap_or(1) as u64);
        }
        Ok(s)
    }

    fn get_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.input.len() { return None; }
        let byte = self.input[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 {
            self.bit_pos = 0;
            self.byte_pos += 1;
        }
        Some(bit)
    }

    fn decode_symbol(&mut self, pdf: &[f64]) -> Result<usize> {
        // Build CDF in [0,1]
        let mut cdf: Vec<f64> = Vec::with_capacity(pdf.len() + 1);
        cdf.push(0.0);
        let mut acc = 0.0f64;
        for &p in pdf { acc += p; cdf.push(acc.min(1.0)); }

        let low_f = self.low as f64;
        let high_f = self.high as f64;
        let range = high_f - low_f + 1.0;
        let scaled = ((self.code - self.low + 1) as f64 - 1.0) / range; // in [0,1)

        // Binary search to find symbol where cdf[sym] <= scaled < cdf[sym+1]
        let mut lo = 0usize;
        let mut hi = pdf.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid + 1] <= scaled { lo = mid + 1; } else if cdf[mid] > scaled { hi = mid; } else { lo = mid; break; }
        }
        let symbol = lo.min(pdf.len() - 1);

        let c_lo = cdf[symbol];
        let c_hi = cdf[symbol + 1];

        self.high = (low_f + (range * c_hi).floor() - 1.0) as u64;
        self.low = (low_f + (range * c_lo).floor()) as u64;

        loop {
            if self.high < self.b_to_pm1 {
                // nothing
            } else if self.low >= self.b_to_pm1 {
                self.low -= self.b_to_pm1;
                self.high -= self.b_to_pm1;
                self.code -= self.b_to_pm1;
            } else if self.low >= self.b_to_pm2 && self.high < self.b_to_pm2 * 3 {
                self.low -= self.b_to_pm2;
                self.high -= self.b_to_pm2;
                self.code -= self.b_to_pm2;
            } else {
                break;
            }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
            self.code = ((self.code << 1) & self.mask) | (self.get_bit().unwrap_or(1) as u64);
        }

        Ok(symbol)
    }
}

// -------------------------------
// Model runner (HF Transformers LLaMA family via safetensors + KV cache).
// -------------------------------

struct LmSession {
    device: Device,
    model: LlamaModel,
    cache: LlamaCache,
    runtime_cfg: LlamaRuntimeConfig,
    dtype: DType,
    vocab_size: usize,
    index_pos: usize, // number of tokens already in cache
}

impl LmSession {
    /// Build a session from weights (one or many .safetensors files) plus the model's config.json.
    fn load(weights_paths: &[PathBuf], config_path: &Path, device: Device) -> Result<Self> {
        // Parse HF LlamaConfig then convert to runtime Config.
        let cfg_bytes = std::fs::read(config_path)
            .with_context(|| format!("failed reading {}", config_path.display()))?;
        let llama_cfg: LlamaConfig = serde_json::from_slice(&cfg_bytes)
            .with_context(|| "failed to parse config.json as HF LlamaConfig")?;
        let runtime_cfg: LlamaRuntimeConfig = llama_cfg.into_config(false /*use_flash_attn*/);

        // Use FP16 for CUDA, FP32 for CPU for better performance
        let dtype = if device.is_cuda() { DType::F16 } else { DType::F32 };

        // Map the safetensors (supports sharded weights).
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(weights_paths, dtype, &device)? };

        // Build model + cache.
        let model = LlamaModel::load(vb.clone(), &runtime_cfg)
            .with_context(|| "failed constructing LLaMA model from safetensors")?;
        let cache = LlamaCache::new(true, dtype, &runtime_cfg, &device)
            .with_context(|| "failed to create KV cache")?;

        Ok(Self {
            device,
            model,
            cache,
            runtime_cfg,
            dtype,
            vocab_size: 0, // set by caller
            index_pos: 0,
        })
    }
    
    /// Get the model's maximum context length from config
    fn max_context_length(&self) -> usize {
        // For robustness on all models, cap to 512 effective steps (encoder uses -1 internally)
        512
    }

    /// Push one token and return logits tensor (device) for the next token.
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

    #[allow(dead_code)]
    fn reset(&mut self) -> Result<()> {
        self.index_pos = 0;
        Ok(())
    }

    /// Reset the KV cache and re-prime it with the provided token history in one batched pass.
    /// Returns the logits for the next token (last step of the history sequence).
    #[allow(dead_code)]
    fn reprime_with_history_and_get_last_logits(&mut self, history: &[u32]) -> Result<Vec<f32>> {
        self.cache = LlamaCache::new(true, self.dtype, &self.runtime_cfg, &self.device)
            .with_context(|| "failed to reset KV cache")?;
        self.index_pos = 0;
        if history.is_empty() {
            bail!("reprime called with empty history");
        }
        let x = Tensor::new(history, &self.device)?.reshape((1, history.len()))?;
        let logits = self.model.forward(&x, self.index_pos, &mut self.cache)?;
        self.index_pos += history.len();
        let v = match logits.rank() {
            1 => logits.to_vec1::<f32>()?,
            2 => {
                logits.i((logits.dim(0)? - 1, ..))?.to_vec1::<f32>()?
            }
            3 => {
                logits.i((0, logits.dim(1)? - 1, ..))?.to_vec1::<f32>()?
            }
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(v)
    }

    /// Reset the KV cache and re-prime it; returns device logits tensor for the next token.
    fn reprime_with_history_and_get_last_logits_tensor(&mut self, history: &[u32]) -> Result<Tensor> {
        self.cache = LlamaCache::new(true, self.dtype, &self.runtime_cfg, &self.device)
            .with_context(|| "failed to reset KV cache")?;
        self.index_pos = 0;
        if history.is_empty() {
            bail!("reprime called with empty history");
        }
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

    /// Reset KV cache and process initial block efficiently for better performance.
    fn reprime_and_process_block(&mut self, history: &[u32], next_block: &[u32]) -> Result<Tensor> {
        // Reset cache
        self.cache = LlamaCache::new(true, self.dtype, &self.runtime_cfg, &self.device)
            .with_context(|| "failed to reset KV cache")?;
        self.index_pos = 0;
        
        // Combine history and next block for one forward pass
        let mut combined = Vec::with_capacity(history.len() + next_block.len());
        combined.extend_from_slice(history);
        combined.extend_from_slice(next_block);
        
        let x = Tensor::new(combined.as_slice(), &self.device)?.reshape((1, combined.len()))?;
        let logits = self.model.forward(&x, self.index_pos, &mut self.cache)?;
        self.index_pos += combined.len();
        
        // Extract logits for predicting next_block tokens
        // logits[i] predicts token at position i+1, so:
        // logits[history.len()-1] predicts next_block[0]
        // logits[history.len()] predicts next_block[1], etc.
        let start_idx = if history.is_empty() { 0 } else { history.len() - 1 };
        let end_idx = start_idx + next_block.len();
        
        let block_logits = match logits.rank() {
            3 => {
                let seq_len = logits.dim(1)?;
                if end_idx > seq_len {
                    bail!("end_idx {} > seq_len {}", end_idx, seq_len);
                }
                logits.i((0, start_idx..end_idx, ..))?
            },
            2 => {
                let seq_len = logits.dim(0)?;
                if end_idx > seq_len {
                    bail!("end_idx {} > seq_len {}", end_idx, seq_len);
                }
                logits.i((start_idx..end_idx, ..))?
            },
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        
        Ok(block_logits)
    }

    /// Process multiple tokens in a single forward pass and return logits for each position.
    /// Returns tensor of shape [block_size, vocab_size] on device.
    fn forward_block_logits(&mut self, block: &[u32]) -> Result<Tensor> {
        let x = Tensor::new(block, &self.device)?.reshape((1, block.len()))?;
        let logits = self.model.forward(&x, self.index_pos, &mut self.cache)?;
        self.index_pos += block.len();
        
        // Normalize to [block_size, vocab_size] regardless of what the model returns
        let out = match logits.rank() {
            3 => logits.i((0, .., ..))?,     // [1, block_size, vocab_size] -> [block_size, vocab_size]
            2 => logits,                     // [block_size, vocab_size]
            1 => {
                // Degenerate single-token case: [vocab_size] -> [1, vocab_size]
                if block.len() != 1 {
                    anyhow::bail!("rank-1 logits but block size > 1: {} tokens", block.len());
                }
                logits.unsqueeze(0)?
            }
            r => anyhow::bail!("unexpected logits rank {}", r),
        };
        Ok(out)
    }

    /// Compute CDF bounds for multiple symbols in a block efficiently with proper floor handling.
    /// Returns vector of (c_lo, c_hi) pairs for each position in the block.
    fn bounds_for_block_on_device(&self, logits_block: &Tensor, symbols: &[usize]) -> Result<Vec<(f64, f64)>> {
        // logits_block is [block_size, vocab_size]
        let block_size = logits_block.dim(0)?;
        let vocab_size = logits_block.dim(1)?;
        
        if symbols.len() != block_size {
            anyhow::bail!("symbol count {} doesn't match block size {}", symbols.len(), block_size);
        }
        
        // Convert to host and apply same logic as decoder for consistency
        let logits_vec = logits_block.to_vec2::<f32>()?;
        let mut bounds = Vec::with_capacity(block_size);
        let p_floor = ac_p_min();
        
        for (pos, &sym) in symbols.iter().enumerate() {
            if sym >= vocab_size {
                anyhow::bail!("symbol {} out of bounds for vocab size {}", sym, vocab_size);
            }
            
            // Apply same softmax + floor logic as decoder
            let pdf = softmax_pdf_floor(&logits_vec[pos], vocab_size, p_floor);
            let p_sym = pdf[sym];
            let c_lo = if sym == 0 { 0.0 } else { pdf[0..sym].iter().sum::<f64>() };
            let c_hi = c_lo + p_sym;
            bounds.push((c_lo, c_hi));
        }
        
        Ok(bounds)
    }

    /// Compute CDF bounds for multiple symbols efficiently in a batch.
    fn compute_bounds_batch(&self, logits_batch: &[Tensor], symbols: &[usize]) -> Result<Vec<(f64, f64)>> {
        if logits_batch.len() != symbols.len() {
            anyhow::bail!("batch size mismatch: {} logits vs {} symbols", logits_batch.len(), symbols.len());
        }
        
        let mut bounds = Vec::with_capacity(logits_batch.len());
        
        for (logits, &sym) in logits_batch.iter().zip(symbols.iter()) {
            let logits_vec = logits.to_vec1::<f32>()?;
            let pdf = softmax_pdf_floor(&logits_vec, self.vocab_size, ac_p_min());
            
            if sym >= pdf.len() {
                anyhow::bail!("symbol {} out of bounds for vocab size {}", sym, pdf.len());
            }
            
            let p_sym = pdf[sym];
            let c_lo = if sym == 0 { 0.0 } else { pdf[0..sym].iter().sum::<f64>() };
            let c_hi = c_lo + p_sym;
            bounds.push((c_lo, c_hi));
        }
        
        Ok(bounds)
    }

    /// Compute CDF bounds for the given symbol using same logic as decoder with comprehensive safety.
    fn bounds_for_symbol_on_device(&self, logits: &Tensor, sym: usize) -> Result<(f64, f64)> {
        // Comprehensive bounds checking
        if sym >= self.vocab_size {
            anyhow::bail!("symbol {} out of bounds for vocab size {}", sym, self.vocab_size);
        }
        
        // Validate tensor dimensions
        let dims = logits.dims();
        if dims.len() != 1 {
            anyhow::bail!("expected 1D logits tensor, got {:?}", dims);
        }
        
        let tensor_len = dims[0];
        if tensor_len != self.vocab_size {
            anyhow::bail!("logits tensor size {} doesn't match vocab size {}", tensor_len, self.vocab_size);
        }
        
        // Convert to host and use same probability computation as decoder
        let logits_vec = match logits.to_vec1::<f32>() {
            Ok(v) => v,
            Err(e) => anyhow::bail!("failed to convert logits to vec: {}", e),
        };
        
        if logits_vec.len() != self.vocab_size {
            anyhow::bail!("logits vec size {} doesn't match vocab size {}", logits_vec.len(), self.vocab_size);
        }
        
        let pdf = softmax_pdf_floor(&logits_vec, self.vocab_size, ac_p_min());
        
        let p_sym = pdf[sym];
        let c_lo = if sym == 0 { 0.0 } else { pdf[0..sym].iter().sum::<f64>() };
        let c_hi = c_lo + p_sym;
        Ok((c_lo, c_hi))
    }
}

// -------------------------------
// Download helpers (HF Hub)
// -------------------------------

/// Download a file from a HF repo (or use the explicit local path if provided).
fn ensure_local_file(repo: &str, file_in_repo: &str, explicit: &Option<PathBuf>) -> Result<PathBuf> {
    if let Some(p) = explicit {
        return Ok(p.clone());
    }
    let api = Api::new()?;
    let r = api.model(repo.to_string());
    let local = r.get(file_in_repo)
        .with_context(|| format!("hf-hub download failed for {repo}/{file_in_repo}. \
                                  If gated, make sure you accepted the license and exported HUGGINGFACE_TOKEN"))?;
    Ok(local)
}

/// Ensure model weights exist locally. Supports:
/// - a single `*.safetensors`
/// - a sharded set described by `model.safetensors.index.json`
/// - the special name `auto` which tries `model.safetensors` then resolves the index.json
///
/// Returns (list_of_weight_paths, repr_filename_for_header, local_config_json).
fn ensure_model_artifacts(repo: &str, model_file: &str, explicit_model: &Option<PathBuf>) -> Result<(Vec<PathBuf>, String, PathBuf)> {
    // Resolve config.json from the model repo (always needed).
    let config_path = ensure_local_file(repo, "config.json", &None)?;

    // Resolve weights
    if let Some(path) = explicit_model {
        let p = path.clone();
        let name_string = p
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("weights")
            .to_string();
        if name_string.ends_with(".safetensors") {
            return Ok((vec![p], name_string, config_path));
        } else if name_string.ends_with(".index.json") {
            let dir = p.parent().unwrap_or_else(|| Path::new("."));
            let bytes = std::fs::read(&p)?;
            let index: serde_json::Value = serde_json::from_slice(&bytes)?;
            let mut files = std::collections::BTreeSet::new();
            if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
                for fname in map.values() {
                    if let Some(f) = fname.as_str() {
                        files.insert(f.to_string());
                    }
                }
            } else {
                bail!("index.json missing weight_map");
            }
            let paths = files.into_iter().map(|f| dir.join(f)).collect::<Vec<_>>();
            return Ok((paths, name_string, config_path));
        } else {
            bail!("--model must point to a .safetensors or .index.json file");
        }
    }

    // Remote resolution via hf-hub
    let api = Api::new()?;
    let repo_api = api.model(repo.to_string());
    let resolve_index = |repo_api: &hf_hub::api::sync::ApiRepo, idx_name: &str| -> Result<Vec<PathBuf>> {
        let idx = repo_api.get(idx_name)?;
        let bytes = std::fs::read(&idx)?;
        let index: serde_json::Value = serde_json::from_slice(&bytes)?;
        let mut files = std::collections::BTreeSet::new();
        if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for fname in map.values() {
                if let Some(f) = fname.as_str() {
                    files.insert(f.to_string());
                }
            }
        } else {
            bail!("index.json missing weight_map");
        }
        let mut out = Vec::new();
        for f in files {
            out.push(repo_api.get(&f)?);
        }
        Ok(out)
    };

    match model_file {
        "auto" => {
            // Try single-file first, then sharded.
            if let Ok(one) = repo_api.get("model.safetensors") {
                Ok((vec![one], "model.safetensors".to_string(), config_path))
            } else {
                let shards = resolve_index(&repo_api, "model.safetensors.index.json")?;
                Ok((shards, "model.safetensors.index.json".to_string(), config_path))
            }
        }
        name if name.ends_with(".safetensors") => {
            let p = repo_api.get(name)?;
            Ok((vec![p], name.to_string(), config_path))
        }
        name if name.ends_with(".index.json") => {
            let shards = resolve_index(&repo_api, name)?;
            Ok((shards, name.to_string(), config_path))
        }
        other => bail!("Unsupported model file '{other}'. Use 'auto', '*.safetensors', or '*.index.json'."),
    }
}

// removed: hex helpers (use binary 16-byte truncation instead)

fn blake3_bytes_bin16(bytes: &[u8]) -> [u8; 16] {
    let hash = blake3::hash(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}

fn blake3_file_bin16(path: &Path) -> Result<[u8; 16]> {
    let mut f = File::open(path)?;
    let mut hasher = Hasher::new();
    let mut buf = [0u8; 1 << 16];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 { break; }
        hasher.update(&buf[..n]);
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(hasher.finalize().as_bytes()[..16].as_ref());
    Ok(out)
}

fn blake3_files_bin16(paths: &[PathBuf]) -> Result<[u8; 16]> {
    let mut hasher = Hasher::new();
    for p in paths {
        if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
            hasher.update(name.as_bytes());
        }
        let mut f = File::open(p)?;
        let mut buf = [0u8; 1 << 16];
        loop {
            let n = f.read(&mut buf)?;
            if n == 0 { break; }
            hasher.update(&buf[..n]);
        }
    }
    let mut out = [0u8; 16];
    out.copy_from_slice(hasher.finalize().as_bytes()[..16].as_ref());
    Ok(out)
}

// -------------------------------
// Encode / Decode
// -------------------------------

fn read_bos_from_config(config_path: &Path) -> Option<u32> {
    if let Ok(bytes) = fs::read(config_path) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(id) = v.get("bos_token_id").and_then(|x| x.as_u64()) {
                return Some(id as u32);
            }
        }
    }
    None
}

fn detect_bos_id(tokenizer: &Tokenizer, config_path: &Path) -> Result<u32> {
    // 1) Prefer bos_token_id from config.json
    if let Some(id) = read_bos_from_config(config_path) {
        return Ok(id);
    }

    // 2) Fallback to common BOS strings in vocab
    let vocab = tokenizer.get_vocab(true);
    let candidates = [
        "<s>",
        "<|bos|>",
        "<|begin_of_text|>",
        "<BOS>",
        "BOS",
        "<bos>",
        "bos",
    ];
    for tok in candidates {
        if let Some(&id) = vocab.get(tok) {
            return Ok(id);
        }
    }

    bail!("could not determine BOS token id from tokenizer")
}

fn encode_file(cli: &Cli, input: &Path, output: &Path) -> Result<()> {
    let t0 = Instant::now();
    let device = detect_device(cli.cpu);
    eprintln!("Device: {}", if device.is_cuda() { "CUDA" } else { "CPU" });

    // Resolve artifacts
    let (weight_paths, weight_repr, config_path) =
        ensure_model_artifacts(&cli.model_repo, &cli.model_file, &cli.model)?;
    let tok_path = ensure_local_file(&cli.tokenizer_repo, &cli.tokenizer_file, &cli.tokenizer)?;

    // Hash files (binary 16-bytes truncation)
    let model_hash16 = blake3_files_bin16(&weight_paths)?;
    let tokenizer_hash16 = blake3_file_bin16(&tok_path)?;

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tok_path.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed loading tokenizer: {e}"))?;

    // Read input
    let data = fs::read(input).with_context(|| "reading input")?;
    let orig_len_bytes = data.len() as u64;
    let orig_blake3_16 = blake3_bytes_bin16(&data);

    // Determine BOS robustly
    let bos_id = detect_bos_id(&tokenizer, &config_path)?;

    // Tokenize WITHOUT auto special tokens; we add BOS ourselves.
    let enc_ns = tokenizer.encode(String::from_utf8_lossy(&data), false)
        .map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
    let mut ids = Vec::<u32>::with_capacity(enc_ns.len() + 1);
    ids.push(bos_id);
    ids.extend(enc_ns.get_ids().iter().copied());
    let token_count = (ids.len() - 1) as u64; // exclude BOS
    let offsets = enc_ns.get_offsets().to_vec();

    // Load model
    let mut session = LmSession::load(&weight_paths, &config_path, device.clone())?;
    let vocab_size = tokenizer.get_vocab_size(true) as usize;
    session.vocab_size = vocab_size;

    // Prepare output
    let mut out = BufWriter::new(File::create(output)?);

    // Build and serialize compact binary header v2
    let header_v2 = HeaderBinV2 {
        bos_token_id: bos_id,
        token_count,
        orig_len_bytes,
        model_hash16: model_hash16,
        tokenizer_hash16: tokenizer_hash16,
        orig_hash16: orig_blake3_16,
        reserved_flags: 0,
        context_window: cli.context as u32,
        vocab_size: vocab_size as u32,
        model_file_repr_len: weight_repr.len() as u32,
        reprime_interval: cli.reprime_interval as u32,
    };
    write_header_v2(&mut out, &header_v2, weight_repr.as_bytes())?;

    // Prepare scan outputs if enabled
    let scan_enabled = cli.scan;
    let scan_dir = cli.scan_output_dir.clone();
    let mut csv_writer_opt: Option<csv::Writer<File>> = None;
    let mut scan_meta: serde_json::Value = json!({});
    if scan_enabled {
        fs::create_dir_all(&scan_dir).ok();
        let csv_path = scan_dir.join("proof.csv");
        let csv_f = File::create(&csv_path)?;
        let mut wtr = csv::Writer::from_writer(csv_f);
        wtr.write_record([
            "file",
            "chunk_index",
            "start_token",
            "end_token",
            "agent_text_len",
            "agent_duration_ms",
            "cross_entropy_baseline_bits",
            "cross_entropy_conditioned_bits",
            "bits_saved",
            "percent_saved",
            "agent_calls",
            "gate",
        ])?;
        wtr.flush()?;
        csv_writer_opt = Some(wtr);

        scan_meta = json!({
            "input_file": input.to_string_lossy(),
            "started_at": Utc::now().to_rfc3339(),
            "model_repo": cli.model_repo,
            "model_file": cli.model_file,
            "tokenizer_repo": cli.tokenizer_repo,
            "tokenizer_file": cli.tokenizer_file,
            "context": cli.context,
            "reprime_interval": cli.reprime_interval,
            "scan": {
                "chunk_size": cli.scan_chunk_size.unwrap_or(cli.reprime_interval),
                "max_steps": cli.scan_max_steps,
                "max_hint_tokens": cli.scan_max_hint_tokens,
                "mcp_config": cli.scan_mcp_config.to_string_lossy(),
                "python": cli.scan_python,
            }
        });
    }

    // AC encode
    let mut ace = ArithmeticEncoder::new(out);

    // Seed with BOS -> logits tensor for next symbol (device)
    let mut _logits_t = session.step_logits_tensor(bos_id)?;
    let mut tokens_since_reprime = 0usize;
    let bar = ProgressBar::new(token_count);
    bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());

        // Optimized encoding with model-aware context management
    // Strict, model-safe context (encoder and decoder mirror this)
    let effective_context = cli.context.min(session.max_context_length().saturating_sub(1));
    
    let scan_chunk = cli.scan_chunk_size.unwrap_or(cli.reprime_interval).max(1);
    // Dedicated scan session to avoid mutating encoder session
    let mut scan_session_opt: Option<LmSession> = if scan_enabled {
        let mut s = LmSession::load(&weight_paths, &config_path, device.clone())?;
        s.vocab_size = vocab_size;
        Some(s)
    } else { None };
    let mut next_scan_boundary: usize = scan_chunk; // in terms of token indices after BOS

    // Aggregated scan metrics
    let mut total_bits_saved: f64 = 0.0;
    let mut total_baseline_bits: f64 = 0.0;
    let mut total_agent_calls: u64 = 0;
    let mut total_chunks: u64 = 0;

    let total_scan_chunks: usize = if scan_enabled { ((token_count as usize) + scan_chunk - 1) / scan_chunk } else { 0 };
    for (i, &sym) in ids.iter().skip(1).enumerate() {
        // Safe reprime strategy that respects model limits
        // Always reprime immediately on hitting the context window.
        // Also reprime if interval budget is hit earlier.
        if session.index_pos >= effective_context && tokens_since_reprime >= cli.reprime_interval {
            let end = 1 + i;
            let start = end.saturating_sub(effective_context);
            let history = &ids[start..end];
            _logits_t = session.reprime_with_history_and_get_last_logits_tensor(history)?;
            tokens_since_reprime = 0;
        }
        
        // Compute interval bounds with optimized probability computation
        let (c_lo, c_hi) = session.bounds_for_symbol_on_device(&_logits_t, sym as usize)?;
        ace.encode_interval(c_lo, c_hi)?;
        
        // Advance model to next step
        _logits_t = session.step_logits_tensor(sym)?;
        tokens_since_reprime += 1;

        // Entropy scan at chunk boundaries
        if scan_enabled && (i + 1) >= next_scan_boundary {
            let chunk_end = 1 + i; // exclusive index in ids slice
            let chunk_start = chunk_end.saturating_sub(scan_chunk);
            // history slice kept for possible future use

            if let Some(wtr) = csv_writer_opt.as_mut() {
                let prefix_end_byte = offsets[(chunk_end - 1).min(offsets.len() - 1)].1 as usize;
                let chunk_index = (chunk_end / scan_chunk).max(1);
                eprintln!("[scan] chunk {}/{} tokens[{}..{}] lookahead={} -> running agent...", chunk_index, total_scan_chunks, chunk_start, chunk_end, cli.scan_lookahead);
                let (agent_text, agent_duration_ms, agent_calls) = run_agent_for_chunk(
                    input,
                    &data,
                    prefix_end_byte,
                    chunk_index,
                    &cli.scan_python,
                    &cli.scan_mcp_config,
                    cli.scan_max_steps,
                    cli.scan_verbose,
                    &scan_dir,
                    cli.scan_agent_timeout,
                )?;
                 eprintln!("[scan] chunk {}/{} agent done in {} ms (calls={}); computing cross-entropy...", chunk_index, total_scan_chunks, agent_duration_ms, agent_calls);
                 if cli.scan_verbose {
                     eprintln!("[scan] agent text length: {} chars, hint tokens budget: {}, lookahead: {}", agent_text.len(), cli.scan_max_hint_tokens, cli.scan_lookahead);
                 }
                 let (baseline_bits, conditioned_bits) = compute_cross_entropy_for_chunk(
                     scan_session_opt.as_mut().expect("scan session"),
                     &tokenizer,
                     bos_id,
                     &ids,
                     chunk_end,
                     &agent_text,
                     cli.scan_max_hint_tokens,
                     cli.scan_lookahead,
                 )?;

                let bits_saved = (baseline_bits - conditioned_bits).max(0.0);
                let percent_saved = if baseline_bits > 0.0 { bits_saved / baseline_bits } else { 0.0 };
                let gate = if bits_saved > 0.0 { 1 } else { 0 };
                total_bits_saved += bits_saved;
                total_baseline_bits += baseline_bits;
                total_agent_calls += agent_calls as u64;
                total_chunks += 1;
                eprintln!("[scan] chunk {}/{} baseline={:.4} bits cond={:.4} bits saved={:.4} ({:.2}%) gate={}", chunk_index, total_scan_chunks, baseline_bits, conditioned_bits, bits_saved, percent_saved * 100.0, gate);

                wtr.write_record(&[
                    input.to_string_lossy().to_string(),
                    chunk_index.to_string(),
                    chunk_start.to_string(),
                    chunk_end.to_string(),
                    agent_text.len().to_string(),
                    agent_duration_ms.to_string(),
                    format!("{:.6}", baseline_bits),
                    format!("{:.6}", conditioned_bits),
                    format!("{:.6}", bits_saved),
                    format!("{:.6}", percent_saved * 100.0),
                    agent_calls.to_string(),
                    gate.to_string(),
                ])?;
                wtr.flush()?;
            }

            next_scan_boundary += scan_chunk;
        }
        
        if i % 1024 == 0 {
            bar.set_position(i as u64);
            let bytes_so_far = ace.bytes_written();
            let bpb = (8.0 * bytes_so_far as f64) / (orig_len_bytes as f64);
            bar.set_message(format!("bytes={}  bpb={:.3}", bytes_so_far, bpb));
        }
    }
    bar.finish_and_clear();

    ace.finish()?;

    // Final evaluation
    let enc_bytes = fs::metadata(output)?.len() as u64;
    let elapsed = t0.elapsed();
    let bpb = (8.0 * enc_bytes as f64) / (orig_len_bytes as f64);
    let char_count = String::from_utf8_lossy(&data).chars().count() as u64;
    let bpc = if char_count > 0 { (8.0 * enc_bytes as f64) / (char_count as f64) } else { f64::NAN };
    println!(
        "Encoded: {} bytes -> {} bytes | bits/byte={:.3} | bits/char={:.3} | context={} | time={:.2?}",
        orig_len_bytes, enc_bytes, bpb, bpc, cli.context, elapsed
    );

    if scan_enabled {
        // finalize meta.json
        let mut meta = scan_meta;
        if let Some(obj) = meta.as_object_mut() {
            obj.insert("orig_len_bytes".to_string(), json!(orig_len_bytes));
            obj.insert("encoded_len_bytes".to_string(), json!(enc_bytes));
            obj.insert("bits_per_byte".to_string(), json!(bpb));
            obj.insert("bits_per_char".to_string(), json!(bpc));
            obj.insert("elapsed_sec".to_string(), json!(elapsed.as_secs_f64()));
            obj.insert("scan_total_chunks".to_string(), json!(total_chunks));
            obj.insert("scan_total_agent_calls".to_string(), json!(total_agent_calls));
            obj.insert("scan_total_bits_saved".to_string(), json!(total_bits_saved));
            let percent_overall = if total_baseline_bits > 0.0 { total_bits_saved / total_baseline_bits } else { 0.0 };
            obj.insert("scan_percent_saved_overall".to_string(), json!(percent_overall * 100.0));
        }
        let meta_path = scan_dir.join("meta.json");
        fs::write(meta_path, serde_json::to_vec_pretty(&meta)?)?;
    }
    Ok(())
}

fn decode_file(cli: &Cli, input: &Path, output: &Path) -> Result<()> {
    let device = detect_device(cli.cpu);
    let mut rdr = BufReader::new(File::open(input)?);
    let (header, model_file_repr) = read_header_v2(&mut rdr)?;

    // Resolve model/tokenizer
    let model_file_str = String::from_utf8_lossy(&model_file_repr).to_string();
    let (weight_paths, _repr, config_path) =
        ensure_model_artifacts(&cli.model_repo, &model_file_str, &cli.model)?;
    let tok_path = ensure_local_file(&cli.tokenizer_repo, &cli.tokenizer_file, &cli.tokenizer)?;

    // Check hashes (unless forced)
    let model_hash16 = blake3_files_bin16(&weight_paths)?;
    let tokenizer_hash16 = blake3_file_bin16(&tok_path)?;
    if !cli.force_mismatch {
        if model_hash16 != header.model_hash16 {
            bail!("model hash mismatch. Use --force-mismatch to override.");
        }
        if tokenizer_hash16 != header.tokenizer_hash16 {
            bail!("tokenizer hash mismatch. Use --force-mismatch to override.");
        }
    }

    // Load tokenizer
    let tokenizer = Tokenizer::from_file(tok_path.to_str().unwrap())
        .map_err(|e| anyhow::anyhow!("failed loading tokenizer: {e}"))?;
    let vocab_size = header.vocab_size as usize;

    // Init model
    let mut session = LmSession::load(&weight_paths, &config_path, device)?;
    session.vocab_size = vocab_size;

    // AC decoder sits on the remainder of the file
    let mut payload = Vec::new();
    rdr.read_to_end(&mut payload)?;
    let mut acd = ArithmeticDecoder::new(&payload[..])?;

    // Seed with BOS -> logits for next (device)
    let mut logits_t = session.step_logits_tensor(header.bos_token_id)?;

    let mut out_tokens: Vec<u32> = Vec::with_capacity(header.token_count as usize + 1);
    out_tokens.push(header.bos_token_id);

    let bar = ProgressBar::new(header.token_count);
    bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());

    let mut tokens_since_reprime = 0usize;
    // For decoding, we still use CPU pdf for arithmetic decoder compatibility.
    let mut logits_vec = logits_t.to_vec1::<f32>()?;
    let mut pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min());
    let effective_context = (header.context_window as usize).saturating_sub(1).min(511);
    for i in 0..header.token_count {
        if session.index_pos >= effective_context && tokens_since_reprime >= header.reprime_interval as usize {
            let end = out_tokens.len();
            let start = end.saturating_sub(effective_context);
            let history = &out_tokens[start..end];
            logits_t = session.reprime_with_history_and_get_last_logits_tensor(history)?;
            tokens_since_reprime = 0;
            logits_vec = logits_t.to_vec1::<f32>()?;
            pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min());
        }
        let sym = acd.decode_symbol(&pdf)? as u32;
        out_tokens.push(sym);
        // Advance model with the decoded symbol
        logits_t = session.step_logits_tensor(sym)?;
        tokens_since_reprime += 1;
        logits_vec = logits_t.to_vec1::<f32>()?;
        pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min());
        if i % 1024 == 0 { bar.set_position(i); }
    }
    bar.finish_and_clear();

    // Convert tokens back to text (skip BOS)
    let detok = tokenizer.decode(&out_tokens[1..], true)
        .map_err(|e| anyhow::anyhow!("tokenizer.decode failed: {e}"))?;
    fs::write(output, detok.as_bytes())?;
    Ok(())
}

// -------------------------------
// Agent subprocess + cross-entropy computation for scan
// -------------------------------

fn pick_python(exe: &str) -> String {
    if exe != "auto" { return exe.to_string(); }
    // Prefer python on PATH; on Windows, 'py -3' can be used but we try 'python' first
    // If this doesn't work: try this "C:\tools\Anaconda3\python.exe"
    "python".to_string()
}

fn run_agent_for_chunk(
    _input_path: &Path,
    full_data: &[u8],
    prefix_end_byte: usize,
    chunk_index: usize,
    python_exe: &str,
    mcp_config: &Path,
    max_steps: usize,
    verbose: bool,
    scan_dir: &Path,
    agent_timeout_secs: u64,
) -> Result<(String, u64, u32)> {
    // * Constructs deterministic prompt describing task and context
    // * Calls agent/agent_cli.py with MCP config and max steps
    // * Streams logs to terminal and to a per-chunk log file

    let prefix = &full_data[..prefix_end_byte.min(full_data.len())];
    let prefix_sample = String::from_utf8_lossy(prefix);
    // Limit prompt size for safety
    let prefix_trunc = if prefix_sample.len() > 8000 { &prefix_sample[prefix_sample.len()-8000..] } else { &prefix_sample };

    let task = format!(
        "You act as a deterministic research agent. Given the observed prefix of a document, use available MCP tools to retrieve information that is likely to appear later in the same document. Prefer authoritative sources. Produce a concise textual synopsis of the likely future content (facts, equations, definitions, datasets, references), suitable for conditioning a language model compressor. Maximize relevance to the document. Do not hallucinate; only include content supported by the tools.\n\nDocument prefix (UTF-8 text):\n{}\n\nOutput: A concise synopsis (plain text).",
        prefix_trunc
    );

    let py = pick_python(python_exe);
    let agent_path = Path::new("agent").join("agent_cli.py");
    let log_path = scan_dir.join(format!("agent_chunk_{}.log", chunk_index));
    // Merge env with agent/.env if present
    let mut env_kv: Vec<(String, String)> = std::env::vars().collect();
    let agent_env_path = Path::new("agent").join(".env");
    if let Ok(bytes) = fs::read(&agent_env_path) {
        if let Ok(text) = String::from_utf8(bytes) {
            for line in text.lines() {
                let s = line.trim();
                if s.is_empty() || s.starts_with('#') { continue; }
                if let Some(eq) = s.find('=') {
                    let (k, v) = s.split_at(eq);
                    let v = &v[1..];
                    env_kv.push((k.trim().to_string(), v.trim().to_string()));
                }
            }
        }
    }

    let mut child = Command::new(py)
        .arg(agent_path)
        .arg("--task").arg(task)
        .arg("--mcp-config").arg(mcp_config)
        .arg("--max-steps").arg(max_steps.to_string())
        .envs(env_kv)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| "failed to spawn agent process")?;

    let mut agent_text = String::new();
    let mut duration_ms: u64 = 0;
    let mut calls: u32 = 0;

    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();

    use std::thread;
    // Reader thread for stdout
    let log_path_out = log_path.clone();
    let t_out = thread::spawn(move || -> (String, u64, u32) {
        let mut out_reader = stdio::BufReader::new(stdout);
        let mut line = String::new();
        let mut local_calls: u32 = 0;
        let mut local_agent_text = String::new();
        let mut local_duration: u64 = 0;
        loop {
            line.clear();
            match out_reader.read_line(&mut line) {
                Ok(n) if n > 0 => {
                    let is_noise = line.contains("DeprecationWarning") || line.contains("PydanticDeprecatedSince") || line.contains("UserWarning") || line.contains("CryptographyDeprecationWarning");
                    if verbose && !is_noise { eprint!("{}", line); }
                    if let Ok(mut f) = File::options().create(true).append(true).open(&log_path_out) {
                        let _ = f.write_all(line.as_bytes());
                    }
                    if let Some(json_start) = line.find("AGENT_RESULT_JSON:") {
                        let j = &line[json_start+"AGENT_RESULT_JSON:".len()..];
                        if let Ok(v) = serde_json::from_str::<serde_json::Value>(j) {
                            if let Some(s) = v.get("final_text").and_then(|x| x.as_str()) { local_agent_text = s.to_string(); }
                            if let Some(d) = v.get("duration_ms").and_then(|x| x.as_u64()) { local_duration = d; }
                        }
                    }
                    if line.contains("Calling tool") || line.contains("Tool:") || line.contains("MCP") || line.contains("Using tool") {
                        local_calls = local_calls.saturating_add(1);
                    }
                }
                _ => break,
            }
        }
        (local_agent_text, local_duration, local_calls)
    });

    // Reader thread for stderr
    let log_path_err = log_path.clone();
    let t_err = thread::spawn(move || -> u32 {
        let mut err_reader = stdio::BufReader::new(stderr);
        let mut line = String::new();
        let mut local_calls: u32 = 0;
        loop {
            line.clear();
            match err_reader.read_line(&mut line) {
                Ok(n) if n > 0 => {
                    let is_noise = line.contains("DeprecationWarning") || line.contains("PydanticDeprecatedSince") || line.contains("UserWarning") || line.contains("CryptographyDeprecationWarning");
                    if verbose && !is_noise { eprint!("{}", line); }
                    if let Ok(mut f) = File::options().create(true).append(true).open(&log_path_err) {
                        let _ = f.write_all(line.as_bytes());
                    }
                    if line.contains("Calling tool") || line.contains("Tool:") || line.contains("MCP") || line.contains("Using tool") {
                        local_calls = local_calls.saturating_add(1);
                    }
                }
                _ => break,
            }
        }
        local_calls
    });

    // Watchdog: timeout and process wait
    let start = std::time::Instant::now();
    let timeout = std::time::Duration::from_secs(if agent_timeout_secs == 0 { 120 } else { agent_timeout_secs });
    loop {
        if let Some(_status) = child.try_wait()? { break; }
        if start.elapsed() >= timeout { break; }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    if child.try_wait()?.is_none() {
        let _ = child.kill();
        eprintln!("[scan] agent timed out after {:?}; killed.", timeout);
    }
    // Ensure readers consume until EOF
    let (text_out, dur_out, calls_out) = t_out.join().unwrap_or((String::new(), 0, 0));
    let calls_err = t_err.join().unwrap_or(0);
    if !text_out.is_empty() { agent_text = text_out; }
    if dur_out > 0 { duration_ms = dur_out; }
    calls = calls.saturating_add(calls_out).saturating_add(calls_err);

    Ok((agent_text, duration_ms, calls))
}

fn tokenize_hint(tokenizer: &Tokenizer, hint: &str, max_tokens: usize) -> Result<Vec<u32>> {
    let enc = tokenizer.encode(hint, false).map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    if ids.len() > max_tokens { ids.truncate(max_tokens); }
    Ok(ids.iter().map(|x| *x as u32).collect())
}

fn compute_cross_entropy_for_chunk(
    session: &mut LmSession,
    tokenizer: &Tokenizer,
    _bos_id: u32,
    all_ids_with_bos: &[u32],
    chunk_end: usize,
    agent_text: &str,
    max_hint_tokens: usize,
    lookahead: usize,
) -> Result<(f64, f64)> {
    // Compute cross-entropy over a bounded local window to keep scan time reasonable.
    // Targets are the next lookahead tokens after chunk_end
    let total_len = all_ids_with_bos.len();
    let end = (chunk_end + lookahead).min(total_len);
    let target_ids = if chunk_end < end { &all_ids_with_bos[chunk_end..end] } else { &[][..] };
    
    if target_ids.is_empty() {
        return Ok((0.0, 0.0)); // No targets to evaluate
    }

    // For baseline: use history up to chunk_end
    let max_ctx = session.max_context_length().saturating_sub(1);
    let history_start = chunk_end.saturating_sub(max_ctx.min(chunk_end));
    let history = &all_ids_with_bos[history_start..chunk_end];
    
    let baseline = cross_entropy_bits_over_span(session, history, target_ids, None)?;

    // For conditioned: tokenize hint and include it in context
    let hint_ids = tokenize_hint(tokenizer, agent_text, max_hint_tokens)?;
    if hint_ids.is_empty() {
        eprintln!("[scan] Warning: No hint tokens generated from agent text (length {})", agent_text.len());
        return Ok((baseline, baseline)); // No hint, same as baseline
    }
    
    let conditioned = cross_entropy_bits_over_span(session, history, target_ids, Some(&hint_ids))?;

    // Debug output to understand the computation
    eprintln!("[scan] XE computation: {} target tokens, {} hint tokens, baseline={:.4} bits, conditioned={:.4} bits, diff={:.4}", 
        target_ids.len(), hint_ids.len(), baseline, conditioned, baseline - conditioned);

    Ok((baseline, conditioned))
}

fn cross_entropy_bits_over_span(
    session: &mut LmSession,
    history: &[u32],
    targets: &[u32],
    hint: Option<&[u32]>,
) -> Result<f64> {
    // Reset cache and prime with context, then compute -log2 p sequentially
    if targets.is_empty() { return Ok(0.0); }

    // Manage context window properly
    let max_ctx = session.max_context_length().saturating_sub(1);
    let mut prime: Vec<u32> = Vec::new();
    
    // Strategy: Place hint early in context, then recent history, so model sees both
    if let Some(h) = hint {
        let hint_budget = h.len().min(max_ctx / 4); // Reserve 3/4 for history
        prime.extend_from_slice(&h[..hint_budget]);
    }
    
    // Add recent history, keeping within context limit
    let remaining_budget = max_ctx.saturating_sub(prime.len());
    let hist_start = history.len().saturating_sub(remaining_budget.min(history.len()));
    prime.extend_from_slice(&history[hist_start..]);
    
    // Truncate if still over limit
    if prime.len() > max_ctx {
        let start = prime.len() - max_ctx;
        prime = prime[start..].to_vec();
    }

    // Prime the model and evaluate targets
    let mut logits_t = session.reprime_with_history_and_get_last_logits_tensor(&prime)?;
    let mut bits: f64 = 0.0;
    
    for &sym in targets.iter() {
        let logits_vec = logits_t.to_vec1::<f32>()?;
        let pdf = softmax_pdf_floor(&logits_vec, session.vocab_size, ac_p_min());
        let p = pdf.get(sym as usize).copied().unwrap_or(ac_p_min());
        bits += -p.max(1e-300).log2();
        logits_t = session.step_logits_tensor(sym)?;
    }
    Ok(bits)
}
// -------------------------------
// Softmax helper
// -------------------------------
fn softmax_pdf_floor(logits: &[f32], vocab_size: usize, p_floor: f64) -> Vec<f64> {
    // Stable softmax -> pdf with floor, then renormalize to sum to 1.
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter().take(vocab_size) {
        if v > max { max = v; }
    }
    let mut exps = vec![0f64; vocab_size];
    let mut sum = 0.0f64;
    for i in 0..vocab_size {
        let e = (logits[i] - max).exp() as f64;
        exps[i] = e;
        sum += e;
    }
    let mut pdf = vec![0f64; vocab_size];
    for i in 0..vocab_size { pdf[i] = (exps[i] / sum).max(p_floor); }
    let norm: f64 = pdf.iter().sum();
    for i in 0..vocab_size { pdf[i] /= norm; }
    pdf
}

// -------------------------------
// Header I/O (binary v2)
// -------------------------------
#[derive(Debug, Clone, Copy)]
struct HeaderBinV2 {
    bos_token_id: u32,
    token_count: u64,
    orig_len_bytes: u64,
    model_hash16: [u8; 16],
    tokenizer_hash16: [u8; 16],
    orig_hash16: [u8; 16],
    reserved_flags: u32,
    context_window: u32,
    vocab_size: u32,
    model_file_repr_len: u32,
    reprime_interval: u32,
}

fn write_var_u64<W: Write>(w: &mut W, mut v: u64) -> Result<()> {
    while v >= 0x80 {
        w.write_all(&[((v as u8) & 0x7F) | 0x80])?;
        v >>= 7;
    }
    w.write_all(&[v as u8])?;
    Ok(())
}

fn read_var_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut shift = 0u32;
    let mut out: u64 = 0;
    loop {
        let mut buf = [0u8; 1];
        r.read_exact(&mut buf)?;
        let byte = buf[0];
        out |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 { break; }
        shift += 7;
        if shift > 63 { bail!("varint too long") }
    }
    Ok(out)
}

fn write_header_v2<W: Write>(w: &mut W, h: &HeaderBinV2, model_file_repr: &[u8]) -> Result<()> {
    w.write_u32::<LittleEndian>(MAGIC)?;
    w.write_u16::<LittleEndian>(VERSION)?;
    // fixed fields
    w.write_u32::<LittleEndian>(h.bos_token_id)?;
    write_var_u64(w, h.token_count)?;
    write_var_u64(w, h.orig_len_bytes)?;
    w.write_all(&h.model_hash16)?;
    w.write_all(&h.tokenizer_hash16)?;
    w.write_all(&h.orig_hash16)?;
    w.write_u32::<LittleEndian>(h.reserved_flags)?;
    w.write_u32::<LittleEndian>(h.context_window)?;
    w.write_u32::<LittleEndian>(h.vocab_size)?;
    w.write_u32::<LittleEndian>(h.model_file_repr_len)?;
    w.write_u32::<LittleEndian>(h.reprime_interval)?;
    // opaque repr payload
    w.write_all(model_file_repr)?;
    Ok(())
}

fn read_header_v2<R: Read>(r: &mut R) -> Result<(HeaderBinV2, Vec<u8>)> {
    let magic = r.read_u32::<LittleEndian>()?;
    if magic != MAGIC { bail!("bad magic") }
    let ver = r.read_u16::<LittleEndian>()?;
    if ver != VERSION { bail!("bad version") }
    let bos_token_id = r.read_u32::<LittleEndian>()?;
    let token_count = read_var_u64(r)?;
    let orig_len_bytes = read_var_u64(r)?;
    let mut model_hash16 = [0u8; 16];
    r.read_exact(&mut model_hash16)?;
    let mut tokenizer_hash16 = [0u8; 16];
    r.read_exact(&mut tokenizer_hash16)?;
    let mut orig_hash16 = [0u8; 16];
    r.read_exact(&mut orig_hash16)?;
    let reserved_flags = r.read_u32::<LittleEndian>()?;
    let context_window = r.read_u32::<LittleEndian>()?;
    let vocab_size = r.read_u32::<LittleEndian>()?;
    let model_file_repr_len = r.read_u32::<LittleEndian>()?;
    let reprime_interval = r.read_u32::<LittleEndian>()?;
    let mut repr = vec![0u8; model_file_repr_len as usize];
    r.read_exact(&mut repr)?;
    Ok((HeaderBinV2 {
        bos_token_id,
        token_count,
        orig_len_bytes,
        model_hash16,
        tokenizer_hash16,
        orig_hash16,
        reserved_flags,
        context_window,
        vocab_size,
        model_file_repr_len,
        reprime_interval,
    }, repr))
}

// -------------------------------
// CLI entry points
// -------------------------------
fn self_test(cli: &Cli, input: &Path) -> Result<()> {
    let tmp_out = input.with_extension("gptz");
    let tmp_dec = input.with_extension("roundtrip.txt");
    let t0 = Instant::now();
    encode_file(cli, input, &tmp_out)?;
    let t1 = Instant::now();
    decode_file(cli, &tmp_out, &tmp_dec)?;
    let t2 = Instant::now();

    let src = fs::read(input)?;
    let enc = fs::read(&tmp_out)?;
    let dec = fs::read(&tmp_dec)?;

    let bits_per_byte = (8.0 * enc.len() as f64) / (src.len() as f64);
    println!("Compression   : {:.2?}", t1 - t0);
    println!("Decompression : {:.2?}", t2 - t1);
    println!("Bits per byte : {:.3}", bits_per_byte);

    if src == dec {
        println!("Roundtrip OK. Encoded: {} bytes, Decoded: {} bytes", enc.len(), dec.len());
    } else {
        bail!("roundtrip mismatch");
    }
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Compress { input, output } => encode_file(&cli, input, output),
        Commands::Decompress { input, output } => decode_file(&cli, input, output),
        Commands::SelfTest { input } => self_test(&cli, input),
    }
}

