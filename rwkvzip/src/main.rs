use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use blake3::Hasher;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::{arg, Parser, Subcommand, ValueHint};
use indicatif::{ProgressBar, ProgressStyle};
use serde::{Deserialize, Serialize};

// Candle
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;

// RWKV7
use candlerwkv7::models::rwkv7::{Config as RwkvConfig, Model as RwkvModel, State as RwkvState, Tokenizer as RwkvTokenizer};

// -------------------------------
// Container format (very small header + AC payload)
// -------------------------------
//
// Magic (u32 LE): 0x5a505447  "RWKZ" (changed from GPTZ)
// Version (u16 LE): 3
// Header Binary V3 (compact binary format)
//
// The binary header has:
// - bos_token_id (u32)
// - token_count (varint u64)
// - orig_len_bytes (varint u64)
// - model_hash16 ([u8; 16])
// - tokenizer_hash16 ([u8; 16])
// - orig_hash16 ([u8; 16])
// - reserved_flags (u32)
// - context_window (u32)
// - vocab_size (u32)
// - model_file_repr_len (u32)
// - reprime_interval (u32)
// - model_file_repr (variable bytes)

const MAGIC: u32 = 0x5a4b5752; // "RWKZ"
const VERSION: u16 = 3;

// Arithmetic coder params (base-2, bitstream output)
const AC_BASE: u32 = 2;
const AC_PRECISION: u32 = 48; // Upgraded from 32 to 48 bits for 65,536x more precision!

// Defaults (local files)
const DEFAULT_MODEL_PATH: &str = "modeldir/prepared/model.safetensors";
const DEFAULT_CONFIG_PATH: &str = "modeldir/prepared/config.json";
const DEFAULT_TOKENIZER_PATH: &str = "modeldir/rwkv_vocab_v20230424.json";

// Model size configurations
fn get_model_paths(size: &str) -> (PathBuf, PathBuf) {
    match size {
        "01" => (
            PathBuf::from("modeldir/prepared/model.safetensors"),
            PathBuf::from("modeldir/prepared/config.json")
        ),
        "04" => (
            PathBuf::from("modeldir/prepared/model_04.safetensors"),
            PathBuf::from("modeldir/prepared/config_04.json")
        ),
        _ => {
            eprintln!("Warning: Unknown size '{}', using default 01", size);
            (
                PathBuf::from("modeldir/prepared/model.safetensors"),
                PathBuf::from("modeldir/prepared/config.json")
            )
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HeaderJson {}

#[derive(Parser, Debug)]
#[command(name="rwkvzip", about="Model-based text compressor using RWKV7 with 64-bit arithmetic coder")]
struct Cli {
    /// Force running on CPU (default tries CUDA 0 then CPU)
    #[arg(long)]
    cpu: bool,

    /// Enable Meta-ICL priming using deterministic, virtual prompts (not transmitted)
    #[arg(long, default_value_t = false)]
    meta_icl: bool,

    /// Interval (in tokens) at which to perform Meta-ICL virtual priming
    #[arg(long, default_value_t = 1024)]
    meta_icl_interval: usize,

    /// Window of recent tokens (count) used in the prompt template (for display only)
    #[arg(long, default_value_t = 512)]
    meta_icl_window: usize,

    /// Max tokens to greedily generate for the virtual summary
    #[arg(long, default_value_t = 24)]
    meta_icl_max_new: usize,

    /// Path to model weights (.safetensors file)
    #[arg(long, default_value = DEFAULT_MODEL_PATH, value_hint = ValueHint::FilePath)]
    model: PathBuf,

    /// Path to config.json file
    #[arg(long, default_value = DEFAULT_CONFIG_PATH, value_hint = ValueHint::FilePath)]
    config: PathBuf,

    /// Path to tokenizer.json (local)
    #[arg(long, default_value = DEFAULT_TOKENIZER_PATH, value_hint = ValueHint::FilePath)]
    tokenizer: PathBuf,

    /// Allow decode even if model/tokenizer BLAKE3 do not match the container
    #[arg(long)]
    force_mismatch: bool,

    /// Context window size (RWKV7 supports infinite context, set to 0 for full sequence)
    #[arg(long, default_value = "0")]
    context: usize,

    /// Reprime interval in tokens (set to 0 to disable repriming for RWKV7 infinite context)
    #[arg(long, default_value = "0")]
    reprime_interval: usize,

    /// Model size variant (01 = 0.1b, 04 = 0.4b)
    #[arg(long, default_value = "01")]
    size: String,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compress a UTF-8 text file into a .rwkz container
    Compress {
        #[arg(value_hint = ValueHint::FilePath)]
        input: PathBuf,
        #[arg(value_hint = ValueHint::FilePath)]
        output: PathBuf,
    },

    /// Decompress a .rwkz container back to a UTF-8 text file
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
    // Adaptive probability floor based on precision and vocabulary size
    let base = AC_BASE as f64;
    let precision = AC_PRECISION as i32;
    // Scale floor based on available precision - more precision = smaller floor
    2.0 * base.powi(-(precision - 8).max(16))
}

fn adaptive_probability_floor(vocab_size: usize, state_entropy: Option<f32>) -> f64 {
    // * Base floor calculation based on vocabulary size
    let base_floor = ac_p_min();
    let vocab_scaling = (vocab_size as f64).log2().max(1.0) / 16.0;
    let mut adjusted_floor = base_floor / vocab_scaling;
    
    // * Adjust based on state entropy if available
    if let Some(entropy) = state_entropy {
        // * Normalize entropy to a reasonable range (0.1 to 2.0)
        let normalized_entropy = entropy.max(0.1).min(2.0);
        
        // * Higher entropy = more uncertainty = use higher floor for conservative coding
        // * Lower entropy = more certainty = use lower floor for aggressive coding
        let entropy_factor = 1.0 + (normalized_entropy as f64 - 1.0) * 0.3;
        adjusted_floor *= entropy_factor;
    }
    
    adjusted_floor
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
// RWKV7 Model Session
// -------------------------------

struct RwkvSession {
    device: Device,
    model: RwkvModel,
    state: RwkvState,
    config: RwkvConfig,
    dtype: DType,
    vocab_size: usize,
    last_state_entropy: Option<f32>,
}

impl RwkvSession {
    /// Build a session from RWKV7 safetensors and config.json
    fn load(model_path: &Path, config_path: &Path, device: Device) -> Result<Self> {
        // Parse RWKV7 config
        let cfg_bytes = std::fs::read(config_path)
            .with_context(|| format!("failed reading {}", config_path.display()))?;
        let config: RwkvConfig = serde_json::from_slice(&cfg_bytes)
            .with_context(|| "failed to parse config.json as RWKV7 Config")?;

        // Use FP32 consistently to avoid dtype mismatches in RWKV operations
        let dtype = DType::F32;

        // Load the safetensors
        let tensors = candle_core::safetensors::load(model_path, &device)
            .with_context(|| format!("failed loading safetensors from {}", model_path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);

        // Build model + state
        let model = RwkvModel::new(&config, vb)
            .with_context(|| "failed constructing RWKV7 model from safetensors")?;
        let state = RwkvState::new(1, &config, None, &device)
            .with_context(|| "failed to create RWKV7 state")?;

        let vocab_size = config.vocab_size;
        Ok(Self {
            device,
            model,
            state,
            config,
            dtype,
            vocab_size,
            last_state_entropy: None,
        })
    }
    
    /// Compute logits for each timestep in the provided sequence (teacher-forced),
    /// returning a tensor of shape (T, V) on device. State is reset before eval.
    fn logits_for_sequence(&mut self, sequence: &[u32]) -> Result<Tensor> {
        if sequence.is_empty() { bail!("empty sequence for logits_for_sequence"); }
        self.state = RwkvState::new(1, &self.config, None, &self.device)
            .with_context(|| "failed to reset RWKV7 state")?;
        // Build (1, T) indices tensor
        let x = Tensor::new(sequence, &self.device)?.reshape((1, sequence.len()))?; // (1, T)
        let logits = self.model.forward(&x, &mut self.state)?; // (1, T, V)
        let t = match logits.rank() {
            3 => logits.i((0, .., ..))?, // (T, V)
            2 => logits,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
    }
    
    /// Get the model's maximum context length from config
    fn max_context_length(&self) -> usize {
        // RWKV7 supports infinite context - return a very large value
        usize::MAX
    }

    /// Push one token and return logits tensor (device) for the next token.
    fn step_logits_tensor(&mut self, token_id: u32) -> Result<Tensor> {
        let x = Tensor::new(&[[token_id]], &self.device)?; // (1, 1)
        let logits = self.model.forward(&x, &mut self.state)?;
        // Extract last logits from (B, T, V) -> (V,)
        let t = match logits.rank() {
            1 => logits,
            2 => logits.i((logits.dim(0)? - 1, ..))?,
            3 => logits.i((0, logits.dim(1)? - 1, ..))?,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
    }



    /// Reset the state and re-prime it with the provided token history in one batched pass.
    /// Returns the logits for the next token (last step of the history sequence).
    fn reprime_with_history_and_get_last_logits_tensor(&mut self, history: &[u32]) -> Result<Tensor> {
        // Reset state
        self.state = RwkvState::new(1, &self.config, None, &self.device)
            .with_context(|| "failed to reset RWKV7 state")?;
        
        if history.is_empty() {
            bail!("reprime called with empty history");
        }
        
        // Process history tokens one by one (RWKV is sequential)
        let mut logits = None;
        for &token in history {
            let x = Tensor::new(&[[token]], &self.device)?;
            logits = Some(self.model.forward(&x, &mut self.state)?);
        }
        
        let logits = logits.unwrap();
        let t = match logits.rank() {
            1 => logits,
            2 => logits.i((logits.dim(0)? - 1, ..))?,
            3 => logits.i((0, logits.dim(1)? - 1, ..))?,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
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
        
        let pdf = softmax_pdf_adaptive(&logits_vec, self.vocab_size, self.last_state_entropy);
        
        let p_sym = pdf[sym];
        let c_lo = if sym == 0 { 0.0 } else { pdf[0..sym].iter().sum::<f64>() };
        let c_hi = c_lo + p_sym;
        Ok((c_lo, c_hi))
    }

    /// Analyze the current RWKV7 state and compute state complexity metrics.
    /// This function computes statistics from the attention state matrices to understand
    /// the patterns encoded in the model's internal state.
    fn advanced_state_analysis_priming(&mut self) -> Result<f32> {
        let mut total_entropy_estimate = 0.0;
        let mut layer_count = 0;

        // Iterate through all layers in the state
        for layer_state in &self.state.per_layer {
            let att_state = &layer_state.att_state;
            
            // Compute entropy-related metrics for this layer's attention state
            let mean = att_state.mean_all()?;
            let centered = att_state.broadcast_sub(&mean)?;
            let variance = centered.sqr()?.mean_all()?;
            let std_dev = variance.sqrt()?;
            
            // Compute entropy estimate based on standard deviation
            // Higher std dev suggests more uncertainty/entropy in the state
            let entropy_estimate = std_dev.to_scalar::<f32>()?;
            total_entropy_estimate += entropy_estimate;
            layer_count += 1;
        }

        if layer_count == 0 {
            return Ok(0.0); // No attention layers found
        }

        // Return average entropy estimate across all layers
        // This represents the overall uncertainty in the model's state
        let avg_entropy = total_entropy_estimate / layer_count as f32;
        Ok(avg_entropy)
    }

    /// Virtually prime the model state with a deterministic prompt and greedy summary.
    /// These tokens are NOT transmitted; both encoder and decoder must call this identically.
    /// Now uses state-aware intelligent triggering based on entropy analysis.
    fn virtual_meta_icl_prime(
        &mut self,
        tokenizer: &RwkvTokenizer,
        _recent_tokens: &[u32],
        window_tokens: usize,
        max_new_tokens: usize,
    ) -> Result<()> {
        // Compute state entropy before any priming
        let current_entropy = self.advanced_state_analysis_priming()?;
        self.last_state_entropy = Some(current_entropy);

        // * Only perform Meta-ICL priming if the state entropy indicates it would be beneficial
        // * High entropy (> 1.0) suggests the model state is uncertain and could benefit from priming
        // * Low entropy (< 0.5) suggests the model state is confident and priming might be disruptive
        if current_entropy < 0.5 {
            // * Skip priming for low-entropy states to avoid disrupting confident predictions
            return Ok(());
        }

        // Deterministic prompt which references window size (purely textual, no content inserted).
        let prompt = format!(
            "\nInstruction: Summarize the last {} tokens in 10 words.\nSummary:",
            window_tokens
        );
        let prompt_ids = tokenizer.encode(&prompt)?;
        let mut last_logits = None;
        for &tok in &prompt_ids {
            last_logits = Some(self.step_logits_tensor(tok)?);
        }
        // Deterministic greedy generation of a small number of tokens.
        if let Some(mut logits) = last_logits {
            for _ in 0..max_new_tokens {
                let v = logits.to_vec1::<f32>()?;
                // argmax
                let mut best_i = 0usize;
                let mut best_v = f32::NEG_INFINITY;
                for (i, &vv) in v.iter().enumerate() {
                    if vv > best_v { best_v = vv; best_i = i; }
                }
                let next_tok = best_i as u32;
                logits = self.step_logits_tensor(next_tok)?;
            }
        }
        Ok(())
    }
}

// -------------------------------
// BLAKE3 helpers
// -------------------------------

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

// -------------------------------
// Encode / Decode
// -------------------------------

fn detect_bos_id(_tokenizer: &RwkvTokenizer, _config_path: &Path) -> Result<u32> {
    // RWKV tokenizer typically uses token 0 as BOS/start
    // We can try to encode common BOS patterns or just use 0
    Ok(0)
}

fn encode_file(cli: &Cli, input: &Path, output: &Path) -> Result<()> {
    let t0 = Instant::now();
    let device = detect_device(cli.cpu);
    eprintln!("Device: {}", if device.is_cuda() { "CUDA" } else { "CPU" });

    // Get model paths based on size parameter
    let (model_path, config_path) = get_model_paths(&cli.size);
    println!("Using model size: {} (model: {}, config: {})", 
             cli.size, model_path.display(), config_path.display());

    // Hash files (binary 16-bytes truncation)
    let model_hash16 = blake3_file_bin16(&model_path)?;
    let tokenizer_hash16 = blake3_file_bin16(&cli.tokenizer)?;

    // Load tokenizer
    let tokenizer = RwkvTokenizer::new(&cli.tokenizer)
        .with_context(|| format!("failed loading tokenizer from {}", cli.tokenizer.display()))?;

    // Read input
    let data = fs::read(input).with_context(|| "reading input")?;
    let orig_len_bytes = data.len() as u64;
    let orig_blake3_16 = blake3_bytes_bin16(&data);

    // Determine BOS robustly
    let bos_id = detect_bos_id(&tokenizer, &cli.config)?;

    // Comprehensive preflight tokenizer roundtrip check
    println!("Performing preflight tokenizer roundtrip check...");
    let mut ids = tokenizer.encode_bytes(&data)
        .with_context(|| "tokenizer.encode_bytes failed during preflight")?;
    let roundtrip_bytes = tokenizer.decode_bytes(&ids);
    
    if roundtrip_bytes != data {
        // Detailed analysis of the mismatch
        let orig_len = data.len();
        let rt_len = roundtrip_bytes.len();
        eprintln!("PREFLIGHT FAILED: Tokenizer roundtrip mismatch detected!");
        eprintln!("Original length: {} bytes", orig_len);
        eprintln!("Roundtrip length: {} bytes", rt_len);
        
        // Find first difference
        let min_len = orig_len.min(rt_len);
        let mut first_diff = None;
        for i in 0..min_len {
            if data[i] != roundtrip_bytes[i] {
                first_diff = Some(i);
                break;
            }
        }
        
        if let Some(pos) = first_diff {
            eprintln!("First byte difference at position {}: orig={:#04x} vs roundtrip={:#04x}", 
                     pos, data[pos], roundtrip_bytes[pos]);
        } else if orig_len != rt_len {
            eprintln!("Length mismatch: original has {} extra bytes", orig_len.abs_diff(rt_len));
        }
        
        // Check if it's a UTF-8 encoding issue
        match String::from_utf8(data.clone()) {
            Ok(_) => eprintln!("Original data is valid UTF-8"),
            Err(e) => eprintln!("Original data has UTF-8 issues at byte {}: {}", e.utf8_error().valid_up_to(), e),
        }
        
        bail!("Tokenizer roundtrip mismatch - aborting to avoid wasted compute time. Check input file encoding.");
    }
    println!("✓ Preflight tokenizer roundtrip check passed");
    
    // Prepend BOS token
    ids.insert(0, bos_id);
    let token_count = (ids.len() - 1) as u64; // exclude BOS

    // Load model
    let mut session = RwkvSession::load(&model_path, &config_path, device)?;

    // Prepare output
    let mut out = BufWriter::new(File::create(output)?);

    // Build and serialize compact binary header v3
    let model_file_repr = model_path.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("model.safetensors")
        .to_string();
    let header_v3 = HeaderBinV3 {
        bos_token_id: bos_id,
        token_count,
        orig_len_bytes,
        model_hash16: model_hash16,
        tokenizer_hash16: tokenizer_hash16,
        orig_hash16: orig_blake3_16,
        reserved_flags: 0,
        context_window: cli.context as u32,
        vocab_size: session.vocab_size as u32,
        model_file_repr_len: model_file_repr.len() as u32,
        reprime_interval: cli.reprime_interval as u32,
    };
    write_header_v3(&mut out, &header_v3, model_file_repr.as_bytes())?;

    // AC encode with optimized RWKV7 infinite context processing
    let mut ace = ArithmeticEncoder::new(out);
    
    // For RWKV7 infinite context: use efficient sequential processing with full context retention
    println!("Encoding with RWKV7 infinite context optimization...");
    
    // Initialize with BOS token
    let mut _logits_t = session.step_logits_tensor(bos_id)?;
    
    // Compute state entropy once for adaptive probability floor
    let current_entropy = session.advanced_state_analysis_priming()?;
    session.last_state_entropy = Some(current_entropy);
    
    let bar = ProgressBar::new(token_count);
    bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len} | {msg}").unwrap());
    let encode_start = Instant::now();
    
    for (i, &sym) in ids.iter().skip(1).enumerate() {
        // Optional Meta-ICL virtual priming (deterministic and symmetric)
        if cli.meta_icl && i > 0 && (i % cli.meta_icl_interval == 0) {
            // Use the already processed portion as "recent" window indicator (we don't include content in prompt).
            // Feeding a fixed prompt to trigger meta-ICL on existing recurrent state.
            session.virtual_meta_icl_prime(&tokenizer, &ids[..=i], cli.meta_icl_window, cli.meta_icl_max_new)?;
            // After priming, obtain fresh logits for the current position by stepping a no-op?
            // Not necessary: we keep using the last logits update scheme below.
            // We will recompute logits for sym with bounds call based on current _logits_t.
        }

        // Compute probability distribution for current token
        let (c_lo, c_hi) = session.bounds_for_symbol_on_device(&_logits_t, sym as usize)?;
        ace.encode_interval(c_lo, c_hi)?;
        
        // Advance model state to next token (RWKV7 automatically maintains infinite context)
        _logits_t = session.step_logits_tensor(sym)?;
        
        if i % 256 == 0 {  // More frequent updates for better UX
            bar.set_position(i as u64);
            let bytes_so_far = ace.bytes_written();
            let bpb = (8.0 * bytes_so_far as f64) / (orig_len_bytes as f64);
            let elapsed = encode_start.elapsed().as_secs_f64();
            let tok_per_sec = if elapsed > 0.0 { (i as f64 + 1.0) / elapsed } else { 0.0 };
            bar.set_message(format!("bytes={}  bpb={:.3}  tok/s={:.1}", bytes_so_far, bpb, tok_per_sec));
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
    let tok_per_sec = if elapsed.as_secs_f64() > 0.0 { (token_count as f64) / elapsed.as_secs_f64() } else { f64::NAN };
    println!(
        "Encoded: {} bytes -> {} bytes | bits/byte={:.3} | bits/char={:.3} | context={} | time={:.2?}",
        orig_len_bytes, enc_bytes, bpb, bpc, cli.context, elapsed
    );
    println!("Throughput: {:.1} tokens/second", tok_per_sec);
    Ok(())
}

fn decode_file(cli: &Cli, input: &Path, output: &Path) -> Result<()> {
    let device = detect_device(cli.cpu);
    let mut rdr = BufReader::new(File::open(input)?);
    let (header, _model_file_repr) = read_header_v3(&mut rdr)?;

    // Get model paths based on size parameter
    let (model_path, config_path) = get_model_paths(&cli.size);

    // Check hashes (unless forced)
    let model_hash16 = blake3_file_bin16(&model_path)?;
    let tokenizer_hash16 = blake3_file_bin16(&cli.tokenizer)?;
    if !cli.force_mismatch {
        if model_hash16 != header.model_hash16 {
            bail!("model hash mismatch. Use --force-mismatch to override.");
        }
        if tokenizer_hash16 != header.tokenizer_hash16 {
            bail!("tokenizer hash mismatch. Use --force-mismatch to override.");
        }
    }

    // Load tokenizer
    let tokenizer = RwkvTokenizer::new(&cli.tokenizer)
        .with_context(|| format!("failed loading tokenizer from {}", cli.tokenizer.display()))?;

    // Init model
    let mut session = RwkvSession::load(&model_path, &config_path, device)?;

    // AC decoder sits on the remainder of the file
    let mut payload = Vec::new();
    rdr.read_to_end(&mut payload)?;
    let mut acd = ArithmeticDecoder::new(&payload[..])?;

    // Seed with BOS -> logits for next (device)
    let mut logits_t = session.step_logits_tensor(header.bos_token_id)?;

    // Compute state entropy once for adaptive probability floor (same as encoder)
    let current_entropy = session.advanced_state_analysis_priming()?;
    session.last_state_entropy = Some(current_entropy);

    let mut out_tokens: Vec<u32> = Vec::with_capacity(header.token_count as usize + 1);
    out_tokens.push(header.bos_token_id);

    let bar = ProgressBar::new(header.token_count);
    bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());

    let mut tokens_since_reprime = 0usize;
    // For decoding, we still use CPU pdf for arithmetic decoder compatibility.
    let mut logits_vec = logits_t.to_vec1::<f32>()?;
            let mut pdf = softmax_pdf_adaptive(&logits_vec, session.vocab_size, session.last_state_entropy);
    
    // RWKV7 infinite context optimization for decoding
    let use_repriming = header.reprime_interval > 0 && header.context_window > 0;
    let effective_context = if header.context_window == 0 { usize::MAX } else { header.context_window as usize };
    
    for i in 0..header.token_count {
        // Optional Meta-ICL virtual priming at the same intervals used by the encoder
        if cli.meta_icl && i > 0 && ((i as usize) % cli.meta_icl_interval == 0) {
            session.virtual_meta_icl_prime(&tokenizer, &out_tokens[..], cli.meta_icl_window, cli.meta_icl_max_new)?;
            // logits_t remains the last state output; next symbol will be decoded below
            logits_vec = logits_t.to_vec1::<f32>()?;
            pdf = softmax_pdf_adaptive(&logits_vec, session.vocab_size, session.last_state_entropy);
        }

        // Only reprime if explicitly configured
        if use_repriming && tokens_since_reprime >= header.reprime_interval as usize {
            let end = out_tokens.len();
            let start = if effective_context == usize::MAX { 0 } else { end.saturating_sub(effective_context) };
            let history = &out_tokens[start..end];
            logits_t = session.reprime_with_history_and_get_last_logits_tensor(history)?;
            tokens_since_reprime = 0;
            logits_vec = logits_t.to_vec1::<f32>()?;
            pdf = softmax_pdf_adaptive(&logits_vec, session.vocab_size, session.last_state_entropy);
        }
        let sym = acd.decode_symbol(&pdf)? as u32;
        out_tokens.push(sym);
        // Advance model with the decoded symbol (RWKV7 maintains infinite context automatically)
        logits_t = session.step_logits_tensor(sym)?;
        tokens_since_reprime += 1;
        logits_vec = logits_t.to_vec1::<f32>()?;
        pdf = softmax_pdf_adaptive(&logits_vec, session.vocab_size, session.last_state_entropy);
        if i % 256 == 0 { bar.set_position(i); }  // More frequent updates
    }
    bar.finish_and_clear();

    // Convert tokens back to raw bytes (skip BOS) for lossless roundtrip
    let detok_bytes = tokenizer.decode_bytes(&out_tokens[1..]);
    fs::write(output, &detok_bytes)?;
    Ok(())
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

// Enhanced softmax with adaptive flooring and better numerical stability
fn softmax_pdf_adaptive(logits: &[f32], vocab_size: usize, state_entropy: Option<f32>) -> Vec<f64> {
    // Use adaptive floor that accounts for vocab size and state entropy
    let p_floor = adaptive_probability_floor(vocab_size, state_entropy);
    softmax_pdf_floor(logits, vocab_size, p_floor)
}

// -------------------------------
// Header I/O (binary v3)
// -------------------------------
#[derive(Debug, Clone, Copy)]
struct HeaderBinV3 {
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

fn write_header_v3<W: Write>(w: &mut W, h: &HeaderBinV3, model_file_repr: &[u8]) -> Result<()> {
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

fn read_header_v3<R: Read>(r: &mut R) -> Result<(HeaderBinV3, Vec<u8>)> {
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
    Ok((HeaderBinV3 {
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
    let tmp_out = input.with_extension("rwkz");
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

