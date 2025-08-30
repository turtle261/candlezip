
// Verbose variants (labels propagate to progress bars)
pub fn compute_compressed_bits_verbose(x: &str, context: Option<&str>, model_path: &std::path::Path, config_path: &std::path::Path, tokenizer_path: &std::path::Path, cpu: bool, proof_mode: bool, label: &str) -> Result<u64> {
    let cfg = NegentConfig { model: model_path.to_path_buf(), config: config_path.to_path_buf(), tokenizer: tokenizer_path.to_path_buf(), cpu };
    let mut ng = Negent::new(&cfg)?;
    let ctx_ids = match context { Some(c) => ng.tokenize_str(c)?, None => Vec::new() };
    let data_ids = ng.tokenize_str(x)?;
    ng.encode_tokens_count_bits(&ctx_ids, &data_ids, proof_mode, Some(label), x.as_bytes().len() as u64)
}

pub fn compute_compressed_bits_bytes_verbose(x: &[u8], context: Option<&[u8]>, model_path: &std::path::Path, config_path: &std::path::Path, tokenizer_path: &std::path::Path, cpu: bool, proof_mode: bool, label: &str) -> Result<u64> {
    let cfg = NegentConfig { model: model_path.to_path_buf(), config: config_path.to_path_buf(), tokenizer: tokenizer_path.to_path_buf(), cpu };
    let mut ng = Negent::new(&cfg)?;
    let ctx_ids = match context { Some(c) => ng.tokenize_bytes(c)?, None => Vec::new() };
    let data_ids = ng.tokenize_bytes(x)?;
    ng.encode_tokens_count_bits(&ctx_ids, &data_ids, proof_mode, Some(label), x.len() as u64)
}

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{bail, Context, Result};
use blake3::Hasher;
use once_cell::sync::Lazy;
use std::time::Instant;

// Candle
use candle_core::{Device, Tensor, DType, IndexOp};
use candle_nn::VarBuilder;
use indicatif::{ProgressBar, ProgressStyle};

// RWKV7
use candlerwkv7::models::rwkv7::{Config as RwkvConfig, Model as RwkvModel, State as RwkvState, Tokenizer as RwkvTokenizer};

// Public defaults (mirror candlezip)
pub const DEFAULT_MODEL_PATH: &str = "modeldir/prepared/model.safetensors";
pub const DEFAULT_CONFIG_PATH: &str = "modeldir/prepared/config.json";
pub const DEFAULT_TOKENIZER_PATH: &str = "modeldir/rwkv_vocab_v20230424.json";

// Arithmetic coder params
const AC_BASE: u32 = 2;
const AC_PRECISION: u32 = 48;

// --- Probability helpers (mirror candlezip) ---
fn ac_p_min() -> f64 {
    let base = AC_BASE as f64;
    let precision = AC_PRECISION as i32;
    2.0 * base.powi(-(precision - 8).max(16))
}

fn adaptive_probability_floor(vocab_size: usize) -> f64 {
    let base_floor = ac_p_min();
    let vocab_scaling = (vocab_size as f64).log2().max(1.0) / 16.0;
    base_floor / vocab_scaling
}

fn softmax_pdf_floor(logits: &[f32], vocab_size: usize, p_floor: f64) -> Vec<f64> {
    let mut max = f32::NEG_INFINITY;
    for &v in logits.iter().take(vocab_size) { if v > max { max = v; } }
    let mut exps = vec![0f64; vocab_size];
    let mut sum = 0.0f64;
    for i in 0..vocab_size { let e = (logits[i] - max).exp() as f64; exps[i] = e; sum += e; }
    let mut pdf = vec![0f64; vocab_size];
    for i in 0..vocab_size { pdf[i] = (exps[i] / sum).max(p_floor); }
    let norm: f64 = pdf.iter().sum();
    for i in 0..vocab_size { pdf[i] /= norm; }
    pdf
}

fn softmax_pdf_adaptive(logits: &[f32], vocab_size: usize) -> Vec<f64> {
    let adaptive_floor = adaptive_probability_floor(vocab_size);
    softmax_pdf_floor(logits, vocab_size, adaptive_floor)
}

// --- Arithmetic coder (bit-precise, base-2) ---
struct ArithmeticEncoder<W: std::io::Write> {
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
    total_bits: u64,
}

impl<W: std::io::Write> ArithmeticEncoder<W> {
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
            total_bits: 0,
        }
    }

    fn write_byte(&mut self, byte: u8) -> Result<()> {
        self.out.write_all(&[byte])?;
        self.bytes_out += 1;
        Ok(())
    }

    fn put_bit_internal(&mut self, bit: u8) -> Result<()> {
        self.bit_buffer = (self.bit_buffer << 1) | (bit & 1);
        self.bit_count += 1;
        self.total_bits += 1;
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
        loop {
            if self.high < self.b_to_pm1 {
                self.put_bit(0)?;
            } else if self.low >= self.b_to_pm1 {
                self.put_bit(1)?;
                self.low -= self.b_to_pm1;
                self.high -= self.b_to_pm1;
            } else if self.low >= self.b_to_pm2 && self.high < self.b_to_pm2 * 3 {
                self.carry_run += 1;
                self.low -= self.b_to_pm2;
                self.high -= self.b_to_pm2;
            } else { break; }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
        }
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.carry_run += 1;
        if self.low < self.b_to_pm2 { self.put_bit(0)?; } else { self.put_bit(1)?; }
        if self.bit_count > 0 {
            let remaining = 8 - self.bit_count;
            for _ in 0..remaining { self.put_bit_internal(0)?; }
        }
        Ok(())
    }

    fn bytes_written(&self) -> u64 { self.bytes_out }
    fn bits_written(&self) -> u64 { self.total_bits }
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
        for _ in 0..precision { s.code = (s.code << 1) | (s.get_bit().unwrap_or(1) as u64); }
        Ok(s)
    }

    fn get_bit(&mut self) -> Option<u8> {
        if self.byte_pos >= self.input.len() { return None; }
        let byte = self.input[self.byte_pos];
        let bit = (byte >> (7 - self.bit_pos)) & 1;
        self.bit_pos += 1;
        if self.bit_pos >= 8 { self.bit_pos = 0; self.byte_pos += 1; }
        Some(bit)
    }

    fn decode_symbol(&mut self, pdf: &[f64]) -> Result<usize> {
        let mut cdf: Vec<f64> = Vec::with_capacity(pdf.len() + 1);
        cdf.push(0.0);
        let mut acc = 0.0f64;
        for &p in pdf { acc += p; cdf.push(acc.min(1.0)); }
        let low_f = self.low as f64;
        let high_f = self.high as f64;
        let range = high_f - low_f + 1.0;
        let scaled = ((self.code - self.low + 1) as f64 - 1.0) / range;
        let mut lo = 0usize; let mut hi = pdf.len();
        while lo < hi {
            let mid = (lo + hi) / 2;
            if cdf[mid + 1] <= scaled { lo = mid + 1; }
            else if cdf[mid] > scaled { hi = mid; }
            else { lo = mid; break; }
        }
        let symbol = lo.min(pdf.len() - 1);
        let c_lo = cdf[symbol]; let c_hi = cdf[symbol + 1];
        self.high = (low_f + (range * c_hi).floor() - 1.0) as u64;
        self.low = (low_f + (range * c_lo).floor()) as u64;
        loop {
            if self.high < self.b_to_pm1 { /* nothing */ }
            else if self.low >= self.b_to_pm1 {
                self.low -= self.b_to_pm1; self.high -= self.b_to_pm1; self.code -= self.b_to_pm1;
            } else if self.low >= self.b_to_pm2 && self.high < self.b_to_pm2 * 3 {
                self.low -= self.b_to_pm2; self.high -= self.b_to_pm2; self.code -= self.b_to_pm2;
            } else { break; }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
            self.code = ((self.code << 1) & self.mask) | (self.get_bit().unwrap_or(1) as u64);
        }
        Ok(symbol)
    }
}

// --- RWKV7 session ---
struct RwkvSession {
    device: Device,
    model: RwkvModel,
    state: RwkvState,
    config: RwkvConfig,
    vocab_size: usize,
}

impl RwkvSession {
    fn load(model_path: &Path, config_path: &Path, device: Device) -> Result<Self> {
        let cfg_bytes = std::fs::read(config_path)
            .with_context(|| format!("failed reading {}", config_path.display()))?;
        let config: RwkvConfig = serde_json::from_slice(&cfg_bytes)
            .with_context(|| "failed to parse config.json as RWKV7 Config")?;
        let dtype = DType::F32;
        let tensors = candle_core::safetensors::load(model_path, &device)
            .with_context(|| format!("failed loading safetensors from {}", model_path.display()))?;
        let vb = VarBuilder::from_tensors(tensors, dtype, &device);
        let model = RwkvModel::new(&config, vb)
            .with_context(|| "failed constructing RWKV7 model from safetensors")?;
        let state = RwkvState::new(1, &config, None, &device)
            .with_context(|| "failed to create RWKV7 state")?;
        let vocab_size = config.vocab_size;
        Ok(Self { device, model, state, config, vocab_size })
    }

    fn reset(&mut self) -> Result<()> {
        self.state = RwkvState::new(1, &self.config, None, &self.device)
            .with_context(|| "failed to reset RWKV7 state")?;
        Ok(())
    }

    fn step_logits_tensor(&mut self, token_id: u32) -> Result<Tensor> {
        let x = Tensor::new(&[[token_id]], &self.device)?;
        let logits = self.model.forward(&x, &mut self.state)?;
        let t = match logits.rank() {
            1 => logits,
            2 => logits.i((logits.dim(0)? - 1, ..))?,
            3 => logits.i((0, logits.dim(1)? - 1, ..))?,
            _ => bail!("unexpected logits shape {:?}", logits.shape()),
        };
        Ok(t)
    }

    fn bounds_for_symbol_on_device(&self, logits: &Tensor, sym: usize) -> Result<(f64, f64)> {
        if sym >= self.vocab_size { bail!("symbol {} out of bounds for vocab size {}", sym, self.vocab_size); }
        let dims = logits.dims();
        if dims.len() != 1 { bail!("expected 1D logits tensor, got {:?}", dims); }
        let tensor_len = dims[0];
        if tensor_len != self.vocab_size { bail!("logits tensor size {} doesn't match vocab size {}", tensor_len, self.vocab_size); }
        let logits_vec = logits.to_vec1::<f32>()?;
        let pdf = softmax_pdf_adaptive(&logits_vec, self.vocab_size);
        let p_sym = pdf[sym];
        let c_lo = if sym == 0 { 0.0 } else { pdf[0..sym].iter().sum::<f64>() };
        let c_hi = c_lo + p_sym;
        Ok((c_lo, c_hi))
    }
}

// --- Hash helpers ---
fn blake3_file_bin16(path: &Path) -> Result<[u8; 16]> {
    let mut f = std::fs::File::open(path)?;
    let mut hasher = Hasher::new();
    let mut buf = [0u8; 1 << 16];
    loop { let n = std::io::Read::read(&mut f, &mut buf)?; if n == 0 { break; } hasher.update(&buf[..n]); }
    let mut out = [0u8; 16];
    out.copy_from_slice(hasher.finalize().as_bytes()[..16].as_ref());
    Ok(out)
}

fn blake3_bytes_bin16(bytes: &[u8]) -> [u8; 16] {
    let hash = blake3::hash(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}

fn detect_device(force_cpu: bool) -> Device {
    if force_cpu { Device::Cpu } else { Device::new_cuda(0).unwrap_or(Device::Cpu) }
}

fn detect_bos_id(_tokenizer: &RwkvTokenizer, _config_path: &Path) -> Result<u32> { Ok(0) }

// --- Session cache keyed by (model_hash16, tokenizer_hash16, device_kind) ---
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct CacheKey {
    model_hash16: [u8; 16],
    tokenizer_hash16: [u8; 16],
    device_cuda: bool,
}

struct CacheEntry {
    tokenizer: std::sync::Arc<RwkvTokenizer>,
    session: RwkvSession,
}

static SESSION_CACHE: Lazy<Mutex<HashMap<CacheKey, CacheEntry>>> = Lazy::new(|| Mutex::new(HashMap::new()));

fn get_or_load_cached_session(model: &Path, config: &Path, tokenizer_path: &Path, cpu: bool) -> Result<(std::sync::Arc<RwkvTokenizer>, RwkvSession)> {
    let model_hash16 = blake3_file_bin16(model)?;
    let tokenizer_hash16 = blake3_file_bin16(tokenizer_path)?;
    let device = detect_device(cpu);
    let key = CacheKey { model_hash16, tokenizer_hash16, device_cuda: device.is_cuda() };
    let mut cache = SESSION_CACHE.lock().unwrap();
    if let Some(entry) = cache.get_mut(&key) {
        // Reset state and return cloned handles
        entry.session.reset()?;
        let tokenizer_arc = entry.tokenizer.clone();
        let session = RwkvSession {
            device: entry.session.device.clone(),
            model: entry.session.model.clone(),
            state: RwkvState::new(1, &entry.session.config, None, &entry.session.device)?,
            config: entry.session.config.clone(),
            vocab_size: entry.session.vocab_size,
        };
        return Ok((tokenizer_arc, session));
    }
    // Load tokenizer and session
    let tokenizer = RwkvTokenizer::new(tokenizer_path)
        .with_context(|| format!("failed loading tokenizer from {}", tokenizer_path.display()))?;
    let tokenizer_arc = std::sync::Arc::new(tokenizer);
    let mut session = RwkvSession::load(model, config, device)?;
    // Put into cache
    cache.insert(key, CacheEntry { tokenizer: tokenizer_arc.clone(), session: RwkvSession {
        device: session.device.clone(),
        model: session.model.clone(),
        state: RwkvState::new(1, &session.config, None, &session.device)?,
        config: session.config.clone(),
        vocab_size: session.vocab_size,
    }});
    Ok((tokenizer_arc, session))
}

// --- Public config & object ---
#[derive(Debug, Clone)]
pub struct NegentConfig {
    pub model: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
    pub cpu: bool,
}

impl Default for NegentConfig {
    fn default() -> Self {
        Self {
            model: PathBuf::from(DEFAULT_MODEL_PATH),
            config: PathBuf::from(DEFAULT_CONFIG_PATH),
            tokenizer: PathBuf::from(DEFAULT_TOKENIZER_PATH),
            cpu: false,
        }
    }
}

pub struct Negent {
    tokenizer: std::sync::Arc<RwkvTokenizer>,
    session: RwkvSession,
    bos_id: u32,
}

impl Negent {
    pub fn new(cfg: &NegentConfig) -> Result<Self> {
        let (tokenizer, mut session) = get_or_load_cached_session(&cfg.model, &cfg.config, &cfg.tokenizer, cfg.cpu)?;
        let bos_id = detect_bos_id(&tokenizer, &cfg.config)?;
        // Prime BOS to warm logits path once (optional)
        session.reset()?;
        Ok(Self { tokenizer, session, bos_id })
    }

    pub fn tokenize_bytes(&self, data: &[u8]) -> Result<Vec<u32>> {
        let ids = self.tokenizer.encode_bytes(data)?;
        // Preflight: decode to ensure lossless
        let roundtrip = self.tokenizer.decode_bytes(&ids);
        if roundtrip != data {
            // Find first difference for diagnostics
            let min_len = roundtrip.len().min(data.len());
            let mut first_diff = None;
            for i in 0..min_len { if roundtrip[i] != data[i] { first_diff = Some(i); break; } }
            if let Some(pos) = first_diff {
                bail!("Tokenizer roundtrip mismatch at byte {}: src={:#04x} != rt={:#04x}", pos, data[pos], roundtrip[pos]);
            } else {
                bail!("Tokenizer roundtrip length mismatch: src={} rt={}", data.len(), roundtrip.len());
            }
        }
        Ok(ids)
    }

    pub fn tokenize_str(&self, s: &str) -> Result<Vec<u32>> { self.tokenize_bytes(s.as_bytes()) }

    fn encode_tokens_count_bits(&mut self, ctx: &[u32], data: &[u32], proof_mode: bool, verbose_label: Option<&str>, orig_len_bytes: u64) -> Result<u64> {
        // Build full sequence with BOS; but only encode 'data' tokens, context is used for priming only
        self.session.reset()?;
        let mut logits_t = self.session.step_logits_tensor(self.bos_id)?;
        // Prime with context tokens (no bits emitted)
        for &sym in ctx.iter() { logits_t = self.session.step_logits_tensor(sym)?; }
        let mut payload: Vec<u8> = Vec::new();
        let mut ace = ArithmeticEncoder::new(&mut payload);

        // Optional progress bar
        let (bar_opt, start_time) = if let Some(lbl) = verbose_label {
            let bar = ProgressBar::new(data.len() as u64);
            bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len} | {msg}").unwrap());
            bar.set_message(format!("{}: start", lbl));
            (Some((lbl.to_string(), bar)), Some(Instant::now()))
        } else { (None, None) };

        for (i, &sym) in data.iter().enumerate() {
            let (c_lo, c_hi) = self.session.bounds_for_symbol_on_device(&logits_t, sym as usize)?;
            ace.encode_interval(c_lo, c_hi)?;
            logits_t = self.session.step_logits_tensor(sym)?;
            if let (Some((ref _lbl, ref bar)), Some(t0)) = (&bar_opt, start_time) {
                if i % 256 == 0 {
                    bar.set_position(i as u64);
                    let bytes_so_far = ace.bytes_written();
                    let bpb = if orig_len_bytes > 0 { (8.0 * bytes_so_far as f64) / (orig_len_bytes as f64) } else { f64::NAN };
                    let elapsed = t0.elapsed().as_secs_f64();
                    let tok_per_sec = if elapsed > 0.0 { (i as f64 + 1.0) / elapsed } else { 0.0 };
                    bar.set_message(format!("bytes={} bpb={:.3} tok/s={:.1}", bytes_so_far, bpb, tok_per_sec));
                }
            }
        }
        if let Some((_, bar)) = &bar_opt { bar.finish_and_clear(); }

        ace.finish()?;
        let bits = (payload.len() as u64) * 8;
        if proof_mode {
            // Decode back and compare
            let mut acd = ArithmeticDecoder::new(&payload[..])?;
            self.session.reset()?;
            let mut logits_t = self.session.step_logits_tensor(self.bos_id)?;
            for &sym in ctx.iter() { logits_t = self.session.step_logits_tensor(sym)?; }
            let mut logits_vec = logits_t.to_vec1::<f32>()?;
            let mut pdf = softmax_pdf_adaptive(&logits_vec, self.session.vocab_size);
            let mut out: Vec<u32> = Vec::with_capacity(data.len());

            // Optional decode progress bar in proof mode
            let (dbar_opt, dstart) = if let Some(lbl) = verbose_label {
                let bar = ProgressBar::new(data.len() as u64);
                bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len} | {msg}").unwrap());
                bar.set_message(format!("{}: decode self-test start", lbl));
                (Some((lbl.to_string(), bar)), Some(Instant::now()))
            } else { (None, None) };

            for i in 0..data.len() {
                let s = acd.decode_symbol(&pdf)? as u32;
                out.push(s);
                logits_t = self.session.step_logits_tensor(s)?;
                logits_vec = logits_t.to_vec1::<f32>()?;
                pdf = softmax_pdf_adaptive(&logits_vec, self.session.vocab_size);
                if out[i] != data[i] { bail!("Proof mode decode mismatch at position {}: got {} expected {}", i, out[i], data[i]); }
                if let (Some((ref _lbl, ref bar)), Some(t0)) = (&dbar_opt, dstart) {
                    if i % 256 == 0 {
                        bar.set_position(i as u64);
                        let elapsed = t0.elapsed().as_secs_f64();
                        let tok_per_sec = if elapsed > 0.0 { (i as f64 + 1.0) / elapsed } else { 0.0 };
                        bar.set_message(format!("decode tok/s={:.1}", tok_per_sec));
                    }
                }
            }
            if let Some((_, bar)) = &dbar_opt { bar.finish_and_clear(); }

            if out != data { bail!("Proof mode mismatch: decoded tokens differ from input"); }
        }
        Ok(bits)
    }

    pub fn compute_compressed_bits_bytes(&mut self, data: &[u8], context: Option<&[u8]>, proof_mode: bool) -> Result<u64> {
        let ctx_ids = match context { Some(c) => self.tokenize_bytes(c)?, None => Vec::new() };
        let data_ids = self.tokenize_bytes(data)?;
        self.encode_tokens_count_bits(&ctx_ids, &data_ids, proof_mode, None, data.len() as u64)
    }

    pub fn compute_compressed_bits_str(&mut self, s: &str, context: Option<&str>, proof_mode: bool) -> Result<u64> {
        let ctx_ids = match context { Some(c) => self.tokenize_str(c)?, None => Vec::new() };
        let data_ids = self.tokenize_str(s)?;
        self.encode_tokens_count_bits(&ctx_ids, &data_ids, proof_mode, None, s.as_bytes().len() as u64)
    }
}

// --- Top-level convenience fns (construct per-call, using cache under the hood) ---
pub fn compute_compressed_bits(x: &str, context: Option<&str>, model_path: &Path, config_path: &Path, tokenizer_path: &Path, cpu: bool, proof_mode: bool) -> Result<u64> {
    let cfg = NegentConfig { model: model_path.to_path_buf(), config: config_path.to_path_buf(), tokenizer: tokenizer_path.to_path_buf(), cpu };
    let mut ng = Negent::new(&cfg)?;
    ng.compute_compressed_bits_str(x, context, proof_mode)
}

pub fn compute_compressed_bits_bytes(x: &[u8], context: Option<&[u8]>, model_path: &Path, config_path: &Path, tokenizer_path: &Path, cpu: bool, proof_mode: bool) -> Result<u64> {
    let cfg = NegentConfig { model: model_path.to_path_buf(), config: config_path.to_path_buf(), tokenizer: tokenizer_path.to_path_buf(), cpu };
    let mut ng = Negent::new(&cfg)?;
    ng.compute_compressed_bits_bytes(x, context, proof_mode)
}

pub fn ncd(x: &str, y: &str, model_path: &Path, config_path: &Path, tokenizer_path: &Path, cpu: bool) -> Result<f64> {
    let cfg = NegentConfig { model: model_path.to_path_buf(), config: config_path.to_path_buf(), tokenizer: tokenizer_path.to_path_buf(), cpu };
    let mut ng = Negent::new(&cfg)?;
    // By default: count bits, no proof to keep it fast
    let c_x = ng.compute_compressed_bits_str(x, None, false)? as f64;
    let c_y = ng.compute_compressed_bits_str(y, None, false)? as f64;
    let c_xy = ng.compute_compressed_bits_str(y, Some(x), false)? as f64; // C(y|x)
    let c_yx = ng.compute_compressed_bits_str(x, Some(y), false)? as f64; // C(x|y)
    Ok((c_xy + c_yx) / (c_x + c_y))
}

pub fn message_distance_bits(x: &str, y: &str, model_path: &Path, config_path: &Path, tokenizer_path: &Path, cpu: bool) -> Result<u64> {
    let cfg = NegentConfig { model: model_path.to_path_buf(), config: config_path.to_path_buf(), tokenizer: tokenizer_path.to_path_buf(), cpu };
    let mut ng = Negent::new(&cfg)?;
    let c_xy = ng.compute_compressed_bits_str(y, Some(x), false)?;
    let c_yx = ng.compute_compressed_bits_str(x, Some(y), false)?;
    Ok(c_xy + c_yx)
}
