use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::process::{Command, Stdio};
use std::io::{self as stdio, BufRead};
use anyhow::{Result, bail, Context};
use blake3::Hasher;
use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use clap::{Parser, Subcommand, ValueHint};
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;
use tokenizers::Tokenizer as HfTokenizer;
use chrono::Utc;
use csv;
use candle_core::{Device, Tensor, DType, IndexOp};

mod models;
use models::{LanguageModelSession, SmolLmSession, Rwkv7Session};

#[derive(Parser, Debug)]
#[command(name="candezip", about="Unified model-based compressor with optional scan mode")]
struct Cli {
    /// Backend: smollm or rwkv7
    #[arg(long, default_value = "smollm")]
    backend: String,

    /// Force CPU
    #[arg(long)]
    cpu: bool,

    // SmolLM (transformer) options
    #[arg(long, value_hint = ValueHint::FilePath)]
    model: Option<PathBuf>,
    #[arg(long, default_value = "HuggingFaceTB/SmolLM2-135M")]
    model_repo: String,
    #[arg(long, default_value = "auto")]
    model_file: String,
    #[arg(long, value_hint = ValueHint::FilePath)]
    tokenizer: Option<PathBuf>,
    #[arg(long, default_value = "HuggingFaceTB/SmolLM2-135M")]
    tokenizer_repo: String,
    #[arg(long, default_value = "tokenizer.json")]
    tokenizer_file: String,

    // RWKV7 options
    #[arg(long, value_hint = ValueHint::FilePath, default_value = "modeldir/prepared/model.safetensors")]
    rwkv_model: PathBuf,
    #[arg(long, value_hint = ValueHint::FilePath, default_value = "modeldir/prepared/config.json")]
    rwkv_config: PathBuf,
    #[arg(long, value_hint = ValueHint::FilePath, default_value = "modeldir/rwkv_vocab_v20230424.json")]
    rwkv_tokenizer: PathBuf,

    /// Context window for transformer path (RWKV7 is effectively infinite)
    #[arg(long, default_value = "512")]
    context: usize,
    /// Reprime interval
    #[arg(long, default_value = "512")]
    reprime_interval: usize,

    /// Enable entropy scan with agent
    #[arg(long, default_value_t = false)]
    scan: bool,
    #[arg(long)]
    scan_chunk_size: Option<usize>,
    #[arg(long, default_value_t = 7)]
    scan_max_steps: usize,
    #[arg(long, default_value_t = 128)]
    scan_max_hint_tokens: usize,
    #[arg(long, value_hint = ValueHint::FilePath, default_value = "agent/mcp_config.json")]
    scan_mcp_config: PathBuf,
    #[arg(long, value_hint = ValueHint::DirPath, default_value = "scan_output")]
    scan_output_dir: PathBuf,
    #[arg(long, default_value = "auto")]
    scan_python: String,
    #[arg(long, default_value_t = false)]
    scan_verbose: bool,
    #[arg(long, default_value_t = 512)]
    scan_lookahead: usize,
    #[arg(long, default_value_t = 120)]
    scan_agent_timeout: u64,

    /// Allow decode even if model/tokenizer BLAKE3 do not match the container
    #[arg(long, default_value_t = false)]
    force_mismatch: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Compress { input: PathBuf, output: PathBuf },
    Decompress { input: PathBuf, output: PathBuf },
    SelfTest { input: PathBuf },
}

fn detect_device(force_cpu: bool) -> candle_core::Device {
    if force_cpu { candle_core::Device::Cpu } else { candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu) }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match &cli.command {
        Commands::Compress { input, output } => encode_file(&cli, input, output),
        Commands::Decompress { input, output } => decode_file(&cli, input, output),
        Commands::SelfTest { input } => self_test(&cli, input),
    }
}

fn self_test(cli: &Cli, input: &Path) -> Result<()> {
    let tmp_out = input.with_extension("canz");
    let tmp_dec = input.with_extension("roundtrip.txt");
    let t0 = std::time::Instant::now();
    encode_file(cli, input, &tmp_out)?;
    let t1 = std::time::Instant::now();
    decode_file(cli, &tmp_out, &tmp_dec)?;
    let t2 = std::time::Instant::now();

    let src = fs::read(input)?;
    let enc = fs::read(&tmp_out)?;
    let dec = fs::read(&tmp_dec)?;

    let bits_per_byte = (8.0 * enc.len() as f64) / (src.len() as f64);
    println!("Compression   : {:.2?}", t1 - t0);
    println!("Decompression : {:.2?}", t2 - t1);
    println!("Bits per byte : {:.3}", bits_per_byte);

    if src == dec {
        println!("Roundtrip OK. Encoded: {} bytes, Decoded: {} bytes", enc.len(), dec.len());
        Ok(())
    } else {
        bail!("roundtrip mismatch");
    }
}

// -------------------------------
// Shared constants/utilities
// -------------------------------

const MAGIC: u32 = 0x5a505447; // "GPTZ"
const VERSION: u16 = 2;

// Arithmetic coder params (base-2, bitstream output)
const AC_BASE: u32 = 2;
const AC_PRECISION: u32 = 32;

// Defaults (HF Hub)
const DEFAULT_MODEL_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const DEFAULT_MODEL_FILE: &str = "auto";
const DEFAULT_TOKENIZER_REPO: &str = "HuggingFaceTB/SmolLM2-135M";
const DEFAULT_TOKENIZER_FILE: &str = "tokenizer.json";

fn ac_p_min() -> f64 {
    let base = AC_BASE as f64;
    2.0 * base.powi(-(AC_PRECISION as i32 - 2))
}

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
        }
    }

    fn write_byte(&mut self, byte: u8) -> anyhow::Result<()> {
        use std::io::Write as _;
        self.out.write_all(&[byte])?;
        self.bytes_out += 1;
        Ok(())
    }

    fn put_bit_internal(&mut self, bit: u8) -> anyhow::Result<()> {
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

    fn put_bit(&mut self, bit: u8) -> anyhow::Result<()> {
        self.put_bit_internal(bit)?;
        while self.carry_run > 0 {
            self.put_bit_internal((!bit) & 1)?;
            self.carry_run -= 1;
        }
        Ok(())
    }

    fn encode_interval(&mut self, c_lo: f64, c_hi: f64) -> anyhow::Result<()> {
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
            } else {
                break;
            }
            self.low = (self.low << 1) & self.mask;
            self.high = ((self.high << 1) & self.mask) | 1;
        }
        Ok(())
    }

    fn finish(mut self) -> anyhow::Result<()> {
        self.carry_run += 1;
        if self.low < self.b_to_pm2 {
            self.put_bit(0)?;
        } else {
            self.put_bit(1)?;
        }
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
    fn new(input: &'a [u8]) -> anyhow::Result<Self> {
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

    fn decode_symbol(&mut self, pdf: &[f64]) -> anyhow::Result<usize> {
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

fn write_var_u64<W: std::io::Write>(w: &mut W, mut v: u64) -> anyhow::Result<()> {
    use std::io::Write as _;
    while v >= 0x80 {
        w.write_all(&[((v as u8) & 0x7F) | 0x80])?;
        v >>= 7;
    }
    w.write_all(&[v as u8])?;
    Ok(())
}

fn read_var_u64<R: std::io::Read>(r: &mut R) -> anyhow::Result<u64> {
    use std::io::Read as _;
    let mut shift = 0u32;
    let mut out: u64 = 0;
    loop {
        let mut buf = [0u8; 1];
        r.read_exact(&mut buf)?;
        let byte = buf[0];
        out |= ((byte & 0x7F) as u64) << shift;
        if (byte & 0x80) == 0 { break; }
        shift += 7;
        if shift > 63 { anyhow::bail!("varint too long"); }
    }
    Ok(out)
}

fn write_header_v2<W: std::io::Write>(w: &mut W, h: &HeaderBinV2, model_file_repr: &[u8]) -> anyhow::Result<()> {
    use byteorder::WriteBytesExt as _;
    w.write_u32::<byteorder::LittleEndian>(MAGIC)?;
    w.write_u16::<byteorder::LittleEndian>(VERSION)?;
    w.write_u32::<byteorder::LittleEndian>(h.bos_token_id)?;
    write_var_u64(w, h.token_count)?;
    write_var_u64(w, h.orig_len_bytes)?;
    w.write_all(&h.model_hash16)?;
    w.write_all(&h.tokenizer_hash16)?;
    w.write_all(&h.orig_hash16)?;
    w.write_u32::<byteorder::LittleEndian>(h.reserved_flags)?;
    w.write_u32::<byteorder::LittleEndian>(h.context_window)?;
    w.write_u32::<byteorder::LittleEndian>(h.vocab_size)?;
    w.write_u32::<byteorder::LittleEndian>(h.model_file_repr_len)?;
    w.write_u32::<byteorder::LittleEndian>(h.reprime_interval)?;
    w.write_all(model_file_repr)?;
    Ok(())
}

fn read_header_v2<R: std::io::Read>(r: &mut R) -> anyhow::Result<(HeaderBinV2, Vec<u8>)> {
    use byteorder::ReadBytesExt as _;
    let magic = r.read_u32::<byteorder::LittleEndian>()?;
    if magic != MAGIC { anyhow::bail!("bad magic"); }
    let ver = r.read_u16::<byteorder::LittleEndian>()?;
    if ver != VERSION { anyhow::bail!("bad version"); }
    let bos_token_id = r.read_u32::<byteorder::LittleEndian>()?;
    let token_count = read_var_u64(r)?;
    let orig_len_bytes = read_var_u64(r)?;
    let mut model_hash16 = [0u8; 16];
    r.read_exact(&mut model_hash16)?;
    let mut tokenizer_hash16 = [0u8; 16];
    r.read_exact(&mut tokenizer_hash16)?;
    let mut orig_hash16 = [0u8; 16];
    r.read_exact(&mut orig_hash16)?;
    let reserved_flags = r.read_u32::<byteorder::LittleEndian>()?;
    let context_window = r.read_u32::<byteorder::LittleEndian>()?;
    let vocab_size = r.read_u32::<byteorder::LittleEndian>()?;
    let model_file_repr_len = r.read_u32::<byteorder::LittleEndian>()?;
    let reprime_interval = r.read_u32::<byteorder::LittleEndian>()?;
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


// ----------------------------------
// HF Hub helpers (SmolLM backend)
// ----------------------------------

fn ensure_local_file(repo: &str, file_in_repo: &str, explicit: &Option<PathBuf>) -> anyhow::Result<PathBuf> {
    if let Some(p) = explicit { return Ok(p.clone()); }
    let api = Api::new()?;
    let r = api.model(repo.to_string());
    let local = r.get(file_in_repo)
        .with_context(|| format!("hf-hub download failed for {repo}/{file_in_repo}. If gated, accept license and export HUGGINGFACE_TOKEN"))?;
    Ok(local)
}

fn ensure_model_artifacts_smol(repo: &str, model_file: &str, explicit_model: &Option<PathBuf>) -> anyhow::Result<(Vec<PathBuf>, String, PathBuf)> {
    let config_path = ensure_local_file(repo, "config.json", &None)?;
    if let Some(path) = explicit_model {
        let p = path.clone();
        let name_string = p.file_name().and_then(|s| s.to_str()).unwrap_or("weights").to_string();
        if name_string.ends_with(".safetensors") {
            return Ok((vec![p], name_string, config_path));
        } else if name_string.ends_with(".index.json") {
            let dir = p.parent().unwrap_or_else(|| Path::new("."));
            let bytes = std::fs::read(&p)?;
            let index: serde_json::Value = serde_json::from_slice(&bytes)?;
            let mut files = std::collections::BTreeSet::new();
            if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
                for fname in map.values() { if let Some(f) = fname.as_str() { files.insert(f.to_string()); } }
            } else { anyhow::bail!("index.json missing weight_map"); }
            let paths = files.into_iter().map(|f| dir.join(f)).collect::<Vec<_>>();
            return Ok((paths, name_string, config_path));
        } else {
            anyhow::bail!("--model must point to a .safetensors or .index.json file");
        }
    }
    let api = Api::new()?; let repo_api = api.model(repo.to_string());
    let resolve_index = |repo_api: &hf_hub::api::sync::ApiRepo, idx_name: &str| -> anyhow::Result<Vec<PathBuf>> {
        let idx = repo_api.get(idx_name)?;
        let bytes = std::fs::read(&idx)?;
        let index: serde_json::Value = serde_json::from_slice(&bytes)?;
        let mut files = std::collections::BTreeSet::new();
        if let Some(map) = index.get("weight_map").and_then(|v| v.as_object()) {
            for fname in map.values() { if let Some(f) = fname.as_str() { files.insert(f.to_string()); } }
        } else { anyhow::bail!("index.json missing weight_map"); }
        let mut out = Vec::new();
        for f in files { out.push(repo_api.get(&f)?); }
        Ok(out)
    };
    match model_file {
        "auto" => {
            if let Ok(one) = repo_api.get("model.safetensors") {
                Ok((vec![one], "model.safetensors".to_string(), config_path))
            } else {
                let shards = resolve_index(&repo_api, "model.safetensors.index.json")?;
                Ok((shards, "model.safetensors.index.json".to_string(), config_path))
            }
        }
        name if name.ends_with(".safetensors") => { let p = repo_api.get(name)?; Ok((vec![p], name.to_string(), config_path)) }
        name if name.ends_with(".index.json") => { let shards = resolve_index(&repo_api, name)?; Ok((shards, name.to_string(), config_path)) }
        other => anyhow::bail!("Unsupported model file '{other}'. Use 'auto', '*.safetensors', or '*.index.json'."),
    }
}

fn read_bos_from_config(config_path: &Path) -> Option<u32> {
    if let Ok(bytes) = fs::read(config_path) {
        if let Ok(v) = serde_json::from_slice::<serde_json::Value>(&bytes) {
            if let Some(id) = v.get("bos_token_id").and_then(|x| x.as_u64()) { return Some(id as u32); }
        }
    }
    None
}

fn detect_bos_id_smol(tokenizer: &HfTokenizer, config_path: &Path) -> anyhow::Result<u32> {
    if let Some(id) = read_bos_from_config(config_path) { return Ok(id); }
    let vocab = tokenizer.get_vocab(true);
    let candidates = ["<s>","<|bos|>","<|begin_of_text|>","<BOS>","BOS","<bos>","bos"];
    for tok in candidates { if let Some(&id) = vocab.get(tok) { return Ok(id); } }
    anyhow::bail!("could not determine BOS token id from tokenizer")
}

fn softmax_pdf_floor(logits: &[f32], vocab_size: usize, p_floor: f64) -> Vec<f64> {
    let mut max = f32::NEG_INFINITY; for &v in logits.iter().take(vocab_size) { if v > max { max = v; } }
    let mut exps = vec![0f64; vocab_size]; let mut sum = 0.0f64;
    for i in 0..vocab_size { let e = (logits[i] - max).exp() as f64; exps[i] = e; sum += e; }
    let mut pdf = vec![0f64; vocab_size];
    for i in 0..vocab_size { pdf[i] = (exps[i] / sum).max(p_floor); }
    let norm: f64 = pdf.iter().sum();
    for i in 0..vocab_size { pdf[i] /= norm; }
    pdf
}

fn bounds_for_symbol_on_device(logits: &candle_core::Tensor, vocab_size: usize, sym: usize) -> anyhow::Result<(f64,f64)> {
    if sym >= vocab_size { anyhow::bail!("symbol {} out of bounds for vocab size {}", sym, vocab_size); }
    let dims = logits.dims(); if dims.len() != 1 { anyhow::bail!("expected 1D logits tensor, got {:?}", dims); }
    let tensor_len = dims[0]; if tensor_len != vocab_size { anyhow::bail!("logits tensor size {} doesn't match vocab size {}", tensor_len, vocab_size); }
    let logits_vec = logits.to_vec1::<f32>()?; let pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min());
    let p_sym = pdf[sym]; let c_lo = if sym==0 { 0.0 } else { pdf[0..sym].iter().sum::<f64>() }; let c_hi = c_lo + p_sym; Ok((c_lo, c_hi))
}

fn combined_pdf_with_literals(logits_vec: &[f32], vocab_size: usize) -> Vec<f64> {
    let mut base = softmax_pdf_floor(logits_vec, vocab_size, ac_p_min());
    let p_escape_total = 256.0 * ac_p_min();
    let scale = (1.0 - p_escape_total).max(0.0);
    for i in 0..vocab_size { base[i] *= scale; }
    let p_literal_each = if p_escape_total > 0.0 { p_escape_total / 256.0 } else { 0.0 };
    let mut pdf = Vec::with_capacity(vocab_size + 256);
    pdf.extend_from_slice(&base);
    for _ in 0..256 { pdf.push(p_literal_each); }
    // renormalize to 1
    let sum: f64 = pdf.iter().sum();
    if sum > 0.0 { for p in pdf.iter_mut() { *p /= sum; } }
    pdf
}

fn cdf_bounds_from_pdf(pdf: &[f64], sym: usize) -> (f64, f64) {
    let mut c_lo = 0.0;
    for i in 0..sym { c_lo += pdf[i]; }
    let c_hi = c_lo + pdf[sym];
    (c_lo, c_hi)
}

fn plan_rwkv_symbols(bytes: &[u8], tok: &candlerwkv7::models::rwkv7::Tokenizer, vocab_size: usize) -> (Vec<u32>, Vec<(usize,usize)>) {
    // For now, disable literal fallback to ensure lossless roundtrip
    // This matches the SmolLM approach of using standard tokenization
    let mut symbols: Vec<u32> = Vec::new();
    let mut offsets: Vec<(usize,usize)> = Vec::new();

    if let Ok(ids) = tok.encode_bytes(bytes) {
        symbols = ids;
        let mut current_offset = 0;
        for &id in &symbols {
            let token_bytes = tok.decode_bytes(&[id]);
            let start = current_offset;
            current_offset += token_bytes.len();
            offsets.push((start, current_offset));
        }
        // Ensure we cover all bytes
        if current_offset < bytes.len() {
            // If tokenizer doesn't cover all bytes, fall back to byte-by-byte
            // but mark as literals for proper handling
            for i in current_offset..bytes.len() {
                symbols.push(vocab_size as u32 + bytes[i] as u32);
                offsets.push((i, i + 1));
            }
        }
    } else {
        // Complete fallback: treat as individual bytes
        for (i, &byte) in bytes.iter().enumerate() {
            symbols.push(vocab_size as u32 + byte as u32);
            offsets.push((i, i + 1));
        }
    }

    (symbols, offsets)
}

fn rwkv_detok_with_literals(tokenizer: &candlerwkv7::models::rwkv7::Tokenizer, symbols: &[u32], vocab_size: usize) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let mut seq: Vec<u32> = Vec::new();

    for &s in symbols {
        if (s as usize) < vocab_size {
            // Regular token
            seq.push(s);
        } else {
            // Literal byte
            if !seq.is_empty() {
                let decoded = tokenizer.decode_bytes(&seq);
                out.extend_from_slice(&decoded);
                seq.clear();
            }
            let byte_value = (s as usize) - vocab_size;
            out.push(byte_value as u8);
        }
    }

    // Handle any remaining sequence
    if !seq.is_empty() {
        let decoded = tokenizer.decode_bytes(&seq);
        out.extend_from_slice(&decoded);
    }

    out
}

// ----------------------------------
// Hash helpers
// ----------------------------------

fn blake3_bytes_bin16(bytes: &[u8]) -> [u8; 16] {
    let hash = blake3::hash(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&hash.as_bytes()[..16]);
    out
}

fn blake3_file_bin16(path: &Path) -> anyhow::Result<[u8; 16]> {
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

fn blake3_files_bin16(paths: &[PathBuf]) -> anyhow::Result<[u8; 16]> {
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

// ----------------------------------
// Scan helpers
// ----------------------------------

fn pick_python(exe: &str) -> String { if exe != "auto" { return exe.to_string(); } "python".to_string() }

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
) -> anyhow::Result<(String, u64, u32)> {
    let prefix = &full_data[..prefix_end_byte.min(full_data.len())];
    let prefix_sample = String::from_utf8_lossy(prefix);
    let prefix_trunc = if prefix_sample.len()>8000 { &prefix_sample[prefix_sample.len()-8000..] } else { &prefix_sample };
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
    if child.try_wait()?.is_none() { let _ = child.kill(); eprintln!("[scan] agent timed out after {:?}; killed.", timeout); }
    let (text_out, dur_out, calls_out) = t_out.join().unwrap_or((String::new(), 0, 0));
    let calls_err = t_err.join().unwrap_or(0);
    if !text_out.is_empty() { agent_text = text_out; }
    if dur_out > 0 { duration_ms = dur_out; }
    calls = calls.saturating_add(calls_out).saturating_add(calls_err);
    Ok((agent_text, duration_ms, calls))
}

fn tokenize_hint_smol(tokenizer: &HfTokenizer, hint: &str, max_tokens: usize) -> anyhow::Result<Vec<u32>> {
    let enc = tokenizer.encode(hint, false).map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
    let mut ids = enc.get_ids().to_vec();
    if ids.len() > max_tokens { ids.truncate(max_tokens); }
    Ok(ids.iter().map(|x| *x as u32).collect())
}

fn tokenize_hint_rwkv(tokenizer: &candlerwkv7::models::rwkv7::Tokenizer, hint: &str, max_tokens: usize) -> anyhow::Result<Vec<u32>> {
    let mut ids = tokenizer.encode(hint)?;
    if ids.len() > max_tokens { ids.truncate(max_tokens); }
    Ok(ids)
}

fn cross_entropy_bits_over_span<S: LanguageModelSession>(session: &mut S, history: &[u32], targets: &[u32], hint: Option<&[u32]>) -> anyhow::Result<f64> {
    if targets.is_empty() { return Ok(0.0); }
    let max_ctx = session.max_context_length().saturating_sub(1);
    let mut prime: Vec<u32> = Vec::new();
    if let Some(h) = hint {
        let hint_budget = h.len().min(max_ctx / 4);
        prime.extend_from_slice(&h[..hint_budget]);
    }
    let remaining_budget = max_ctx.saturating_sub(prime.len());
    let hist_start = history.len().saturating_sub(remaining_budget.min(history.len()));
    prime.extend_from_slice(&history[hist_start..]);
    if prime.len() > max_ctx { let start = prime.len() - max_ctx; prime = prime[start..].to_vec(); }
    let mut logits_t = session.reprime_with_history_and_get_last_logits_tensor(&prime)?;
    let mut bits: f64 = 0.0;
    let vocab = session.vocab_size();
    for &sym in targets.iter() {
        let logits_vec = logits_t.to_vec1::<f32>()?;
        let pdf = softmax_pdf_floor(&logits_vec, vocab, ac_p_min());
        let p = pdf.get(sym as usize).copied().unwrap_or(ac_p_min());
        bits += -p.max(1e-300).log2();
        logits_t = session.step_logits_tensor(sym)?;
    }
    Ok(bits)
}

fn compute_cross_entropy_for_chunk_smol(
    session: &mut SmolLmSession,
    tokenizer: &HfTokenizer,
    _bos_id: u32,
    all_ids_with_bos: &[u32],
    chunk_end: usize,
    agent_text: &str,
    max_hint_tokens: usize,
    lookahead: usize,
) -> anyhow::Result<(f64, f64)> {
    let total_len = all_ids_with_bos.len();
    let end = (chunk_end + lookahead).min(total_len);
    let target_ids = if chunk_end < end { &all_ids_with_bos[chunk_end..end] } else { &[][..] };
    if target_ids.is_empty() { return Ok((0.0, 0.0)); }
    let max_ctx = session.max_context_length().saturating_sub(1);
    let history_start = chunk_end.saturating_sub(max_ctx.min(chunk_end));
    let history = &all_ids_with_bos[history_start..chunk_end];
    let baseline = cross_entropy_bits_over_span(session, history, target_ids, None)?;
    let hint_ids = tokenize_hint_smol(tokenizer, agent_text, max_hint_tokens)?;
    if hint_ids.is_empty() { eprintln!("[scan] Warning: No hint tokens generated from agent text (length {})", agent_text.len()); return Ok((baseline, baseline)); }
    let conditioned = cross_entropy_bits_over_span(session, history, target_ids, Some(&hint_ids))?;
    eprintln!("[scan] XE computation: {} target tokens, {} hint tokens, baseline={:.4} bits, conditioned={:.4} bits, diff={:.4}", target_ids.len(), hint_ids.len(), baseline, conditioned, baseline - conditioned);
    Ok((baseline, conditioned))
}

fn compute_cross_entropy_for_chunk_rwkv(
    session: &mut Rwkv7Session,
    tokenizer: &candlerwkv7::models::rwkv7::Tokenizer,
    _bos_id: u32,
    all_ids_with_bos: &[u32],
    chunk_end: usize,
    agent_text: &str,
    max_hint_tokens: usize,
    lookahead: usize,
) -> anyhow::Result<(f64, f64)> {
    let total_len = all_ids_with_bos.len();
    let end = (chunk_end + lookahead).min(total_len);
    let target_ids = if chunk_end < end { &all_ids_with_bos[chunk_end..end] } else { &[][..] };
    if target_ids.is_empty() { return Ok((0.0, 0.0)); }
    let history = &all_ids_with_bos[0..chunk_end];
    let baseline = cross_entropy_bits_over_span(session, history, target_ids, None)?;
    let hint_ids = tokenize_hint_rwkv(tokenizer, agent_text, max_hint_tokens)?;
    if hint_ids.is_empty() { eprintln!("[scan] Warning: No hint tokens generated from agent text (length {})", agent_text.len()); return Ok((baseline, baseline)); }
    let conditioned = cross_entropy_bits_over_span(session, history, target_ids, Some(&hint_ids))?;
    eprintln!("[scan] XE computation: {} target tokens, {} hint tokens, baseline={:.4} bits, conditioned={:.4} bits, diff={:.4}", target_ids.len(), hint_ids.len(), baseline, conditioned, baseline - conditioned);
    Ok((baseline, conditioned))
}

// ----------------------------------
// Encode / Decode (unified)
// ----------------------------------

fn encode_file(cli: &Cli, input: &Path, output: &Path) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    let device = detect_device(cli.cpu);
    eprintln!("Device: {}", if device.is_cuda() { "CUDA" } else { "CPU" });

    let backend = cli.backend.to_lowercase();
    let (mut session_smol, mut session_rwkv): (Option<SmolLmSession>, Option<Rwkv7Session>) = (None,None);
    let mut vocab_size: usize = 0;
    let mut bos_id: u32 = 0;
    let mut tok_hf: Option<HfTokenizer> = None;
    let mut tok_rwkv: Option<candlerwkv7::models::rwkv7::Tokenizer> = None;
    let (model_hash16, tokenizer_hash16, weight_repr, config_path): ([u8;16],[u8;16], String, PathBuf);

    if backend=="smollm" {
        let (weight_paths, weight_repr_s, config_p) = ensure_model_artifacts_smol(&cli.model_repo, &cli.model_file, &cli.model)?;
        let tok_path = ensure_local_file(&cli.tokenizer_repo, &cli.tokenizer_file, &cli.tokenizer)?;
        model_hash16 = blake3_files_bin16(&weight_paths)?;
        tokenizer_hash16 = blake3_file_bin16(&tok_path)?;
        let tokenizer = HfTokenizer::from_file(tok_path.to_str().unwrap()).map_err(|e| anyhow::anyhow!("failed loading tokenizer: {e}"))?;
        bos_id = detect_bos_id_smol(&tokenizer, &config_p)?;
        vocab_size = tokenizer.get_vocab_size(true) as usize;
        let s = SmolLmSession::load(&weight_paths, &config_p, device.clone(), vocab_size)?;
        session_smol = Some(s); tok_hf = Some(tokenizer); weight_repr = weight_repr_s; config_path = config_p;
    } else if backend=="rwkv7" {
        let model_path = cli.rwkv_model.clone();
        let config_p = cli.rwkv_config.clone();
        let tok_path = cli.rwkv_tokenizer.clone();
        model_hash16 = blake3_file_bin16(&model_path)?;
        tokenizer_hash16 = blake3_file_bin16(&tok_path)?;
        let tokenizer = candlerwkv7::models::rwkv7::Tokenizer::new(&tok_path)?;
        bos_id = 0;
        let s = Rwkv7Session::load(&model_path, &config_p, device.clone())?;
        vocab_size = s.vocab_size();
        session_rwkv = Some(s); tok_rwkv = Some(tokenizer); weight_repr = model_path.file_name().and_then(|s| s.to_str()).unwrap_or("model.safetensors").to_string(); config_path = config_p;
    } else { anyhow::bail!("unknown backend {}", backend); }

    // Read input
    let data = fs::read(input).with_context(|| "reading input")?; let orig_len_bytes = data.len() as u64; let orig_blake3_16 = blake3_bytes_bin16(&data);

    // Tokenize - match _gptzip implementation exactly
    let (mut ids, offsets): (Vec<u32>, Vec<(usize,usize)>) = if let Some(ref tok)=tok_hf {
        // SmolLM: Use standard HF tokenizer encoding (matches _gptzip)
        let enc_ns = tok.encode(String::from_utf8_lossy(&data), false)
            .map_err(|e| anyhow::anyhow!("tokenizer.encode failed: {e}"))?;
        let mut v = Vec::<u32>::with_capacity(enc_ns.len() + 1);
        v.push(bos_id);
        v.extend(enc_ns.get_ids().iter().copied().map(|x| x as u32));
        (v, enc_ns.get_offsets().to_vec())
    } else {
        // RWKV7: Use planned tokenization with fallback
        let tok = tok_rwkv.as_ref().unwrap();
        let (mut planned, offs) = plan_rwkv_symbols(&data, tok, vocab_size);
        planned.insert(0, bos_id);
        (planned, offs)
    };
    let token_count = (ids.len() - 1) as u64;

    // Prepare output and header
    let mut out = BufWriter::new(File::create(output)?);
    let header_v2 = HeaderBinV2 { bos_token_id: bos_id, token_count, orig_len_bytes, model_hash16, tokenizer_hash16, orig_hash16: orig_blake3_16, reserved_flags: 0, context_window: cli.context as u32, vocab_size: vocab_size as u32, model_file_repr_len: weight_repr.len() as u32, reprime_interval: cli.reprime_interval as u32 };
    write_header_v2(&mut out, &header_v2, weight_repr.as_bytes())?;

    // Scan setup
    let scan_enabled = cli.scan; let mut scan_dir = cli.scan_output_dir.clone(); let mut csv_writer_opt: Option<csv::Writer<File>> = None; let mut scan_meta: serde_json::Value = json!({});
    if scan_enabled {
        // Per-run dir to prevent cross-run contamination and logs overlap
        let ts = Utc::now().format("%Y%m%d_%H%M%S");
        let run_dir_name = format!("{}_{}_{}", input.file_stem().and_then(|s| s.to_str()).unwrap_or("input"), backend, ts);
        scan_dir = scan_dir.join(run_dir_name);
        fs::create_dir_all(&scan_dir).ok();
        let csv_path=scan_dir.join("proof.csv"); let csv_f=File::create(&csv_path)?; let mut wtr = csv::Writer::from_writer(csv_f);
        wtr.write_record(["file","chunk_index","start_token","end_token","agent_text_len","agent_duration_ms","cross_entropy_baseline_bits","cross_entropy_conditioned_bits","bits_saved","percent_saved","agent_calls","gate"]) ?; wtr.flush()?; csv_writer_opt=Some(wtr);
        scan_meta = json!({"input_file": input.to_string_lossy(), "started_at": Utc::now().to_rfc3339(), "backend": backend, "context": cli.context, "reprime_interval": cli.reprime_interval, "scan": {"chunk_size": cli.scan_chunk_size.unwrap_or(cli.reprime_interval), "max_steps": cli.scan_max_steps, "max_hint_tokens": cli.scan_max_hint_tokens, "mcp_config": cli.scan_mcp_config.to_string_lossy(), "python": cli.scan_python }, "run_dir": scan_dir.to_string_lossy() });
    }

    // AC encode
    let mut ace = ArithmeticEncoder::new(out);
    let mut logits_t = if backend=="smollm" { session_smol.as_mut().unwrap().step_logits_tensor(bos_id)? } else { session_rwkv.as_mut().unwrap().step_logits_tensor(bos_id)? };
    let mut tokens_since_reprime = 0usize;
    let bar = ProgressBar::new(token_count);
    bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());
    // Backend-specific tuning: SmolLM capped context; RWKV7 can use very long histories
    let effective_context = if backend=="smollm" { cli.context.min(511) } else { usize::MAX };
    let scan_chunk = if backend=="rwkv7" {
        // Prefer larger chunks for RWKV7 to leverage infinite context
        cli.scan_chunk_size.unwrap_or(2048).max(512)
    } else { cli.scan_chunk_size.unwrap_or(cli.reprime_interval).max(1) };
    let mut next_scan_boundary = scan_chunk;
    let mut total_bits_saved=0.0f64; let mut total_baseline_bits=0.0f64; let mut total_agent_calls: u64=0; let mut total_chunks: u64=0;
    let total_scan_chunks: usize = if scan_enabled { ((token_count as usize) + scan_chunk - 1) / scan_chunk } else { 0 };
    // Separate scan sessions to avoid state interference
    let mut scan_session_smol: Option<SmolLmSession> = None;
    let mut scan_session_rwkv: Option<Rwkv7Session> = None;
    if scan_enabled {
        if backend=="smollm" {
            let (weight_paths, _repr, config_p) = ensure_model_artifacts_smol(&cli.model_repo, &cli.model_file, &cli.model)?;
            let tokenizer_path = ensure_local_file(&cli.tokenizer_repo, &cli.tokenizer_file, &cli.tokenizer)?;
            let tokenizer = HfTokenizer::from_file(tokenizer_path.to_str().unwrap()).map_err(|e| anyhow::anyhow!("failed loading tokenizer: {e}"))?;
            let s = SmolLmSession::load(&weight_paths, &config_p, device.clone(), tokenizer.get_vocab_size(true))?;
            scan_session_smol = Some(s);
        } else {
            let s = Rwkv7Session::load(&cli.rwkv_model, &cli.rwkv_config, device.clone())?;
            scan_session_rwkv = Some(s);
        }
    }

    for (i, &sym) in ids.iter().skip(1).enumerate() {
        // Reprime logic - exactly match _gptzip implementation
        if backend=="smollm" {
            // SmolLM: Reprime when session.index_pos >= effective_context AND tokens_since_reprime >= reprime_interval
            let session = session_smol.as_ref().unwrap();
            if session.index_pos() >= effective_context && tokens_since_reprime >= cli.reprime_interval {
                let end = 1 + i;
                let start = end.saturating_sub(effective_context);
                let history = &ids[start..end];
                logits_t = session_smol.as_mut().unwrap().reprime_with_history_and_get_last_logits_tensor(history)?;
                tokens_since_reprime = 0;
            }
        }

        let (c_lo, c_hi) = if backend=="smollm" {
            session_smol.as_ref().unwrap().bounds_for_symbol_on_device(&logits_t, sym as usize)?
        } else {
            let logits_vec = logits_t.to_vec1::<f32>()?;
            let pdf = combined_pdf_with_literals(&logits_vec, vocab_size);
            cdf_bounds_from_pdf(&pdf, sym as usize)
        };
        ace.encode_interval(c_lo, c_hi)?;

        if backend=="smollm" {
            logits_t = session_smol.as_mut().unwrap().step_logits_tensor(sym)?;
        } else {
            if (sym as usize) < vocab_size {
                logits_t = session_rwkv.as_mut().unwrap().step_logits_tensor(sym)?;
            }
        }
        tokens_since_reprime += 1;

        if scan_enabled && (i + 1) >= next_scan_boundary {
            let chunk_end = 1 + i; let chunk_start = chunk_end.saturating_sub(scan_chunk);
            if let Some(wtr) = csv_writer_opt.as_mut() {
                let prefix_end_byte = if offsets.is_empty() { chunk_end } else { offsets[(chunk_end - 1).min(offsets.len() - 1)].1 as usize };
                let chunk_index = (chunk_end / scan_chunk).max(1);
                eprintln!("[scan] [{}] chunk {}/{} tokens[{}..{}] lookahead={} -> agent...", backend, chunk_index, total_scan_chunks, chunk_start, chunk_end, cli.scan_lookahead);
                // RWKV-specific: increase allowed steps modestly if default and backend is rwkv7
                let max_steps = if backend=="rwkv7" && cli.scan_max_steps == 7 { 10 } else { cli.scan_max_steps };
                let (agent_text, agent_duration_ms, agent_calls) = run_agent_for_chunk(input, &data, prefix_end_byte, chunk_index, &cli.scan_python, &cli.scan_mcp_config, max_steps, cli.scan_verbose, &scan_dir, cli.scan_agent_timeout)?;
                eprintln!("[scan] [{}] chunk {}/{} agent done in {} ms (calls={}), computing XE...", backend, chunk_index, total_scan_chunks, agent_duration_ms, agent_calls);
                let (baseline_bits, conditioned_bits) = if backend=="smollm" { compute_cross_entropy_for_chunk_smol(scan_session_smol.as_mut().unwrap(), tok_hf.as_ref().unwrap(), bos_id, &ids, chunk_end, &agent_text, cli.scan_max_hint_tokens, cli.scan_lookahead)? } else { compute_cross_entropy_for_chunk_rwkv(scan_session_rwkv.as_mut().unwrap(), tok_rwkv.as_ref().unwrap(), bos_id, &ids, chunk_end, &agent_text, cli.scan_max_hint_tokens, cli.scan_lookahead)? };
                let bits_saved=(baseline_bits-conditioned_bits).max(0.0); let percent_saved=if baseline_bits>0.0 { bits_saved / baseline_bits } else { 0.0 }; let gate= if bits_saved>0.0 { 1 } else { 0 };
                total_bits_saved += bits_saved; total_baseline_bits += baseline_bits; total_agent_calls += agent_calls as u64; total_chunks += 1;
                eprintln!("[scan] chunk {}/{} baseline={:.4} bits cond={:.4} bits saved={:.4} ({:.2}%) gate={}", chunk_index, total_scan_chunks, baseline_bits, conditioned_bits, bits_saved, percent_saved*100.0, gate);
                wtr.write_record(&[ input.to_string_lossy().to_string(), chunk_index.to_string(), chunk_start.to_string(), chunk_end.to_string(), agent_text.len().to_string(), agent_duration_ms.to_string(), format!("{:.6}", baseline_bits), format!("{:.6}" , conditioned_bits), format!("{:.6}", bits_saved), format!("{:.6}", percent_saved*100.0), agent_calls.to_string(), gate.to_string() ])?; wtr.flush()?;
            }
            next_scan_boundary += scan_chunk;
        }
        if i % 1024 == 0 { bar.set_position(i as u64); let bytes_so_far = ace.bytes_written(); let bpb = (8.0 * bytes_so_far as f64) / (orig_len_bytes as f64); bar.set_message(format!("bytes={}  bpb={:.3}", bytes_so_far, bpb)); }
    }
    bar.finish_and_clear(); ace.finish()?;

    let enc_bytes = fs::metadata(output)?.len() as u64; let elapsed = t0.elapsed(); let bpb = (8.0 * enc_bytes as f64) / (orig_len_bytes as f64); let char_count = String::from_utf8_lossy(&data).chars().count() as u64; let bpc = if char_count > 0 { (8.0 * enc_bytes as f64) / (char_count as f64) } else { f64::NAN };
    println!("Encoded: {} bytes -> {} bytes | bits/byte={:.3} | bits/char={:.3} | context={} | time={:.2?}", orig_len_bytes, enc_bytes, bpb, bpc, cli.context, elapsed);
    if scan_enabled { let mut meta=scan_meta; if let Some(obj)=meta.as_object_mut() { obj.insert("orig_len_bytes".to_string(), json!(orig_len_bytes)); obj.insert("encoded_len_bytes".to_string(), json!(enc_bytes)); obj.insert("bits_per_byte".to_string(), json!(bpb)); obj.insert("bits_per_char".to_string(), json!(bpc)); obj.insert("elapsed_sec".to_string(), json!(elapsed.as_secs_f64())); obj.insert("scan_total_chunks".to_string(), json!(total_chunks)); obj.insert("scan_total_agent_calls".to_string(), json!(total_agent_calls)); obj.insert("scan_total_bits_saved".to_string(), json!(total_bits_saved)); let percent_overall = if total_baseline_bits>0.0 { total_bits_saved/total_baseline_bits } else { 0.0 }; obj.insert("scan_percent_saved_overall".to_string(), json!(percent_overall*100.0)); } let meta_path=scan_dir.join("meta.json"); fs::write(meta_path, serde_json::to_vec_pretty(&meta)?)?; }
    Ok(())
}

fn decode_file(cli: &Cli, input: &Path, output: &Path) -> anyhow::Result<()> {
    let device = detect_device(cli.cpu);
    let mut rdr = BufReader::new(File::open(input)?);
    let (header, model_file_repr) = read_header_v2(&mut rdr)?;
    let backend = cli.backend.to_lowercase();
    let vocab_size = header.vocab_size as usize;

    if backend=="smollm" {
        let model_file_str = String::from_utf8_lossy(&model_file_repr).to_string();
        let (weight_paths, _repr, config_path) = ensure_model_artifacts_smol(&cli.model_repo, &model_file_str, &cli.model)?;
        let tok_path = ensure_local_file(&cli.tokenizer_repo, &cli.tokenizer_file, &cli.tokenizer)?;
        let model_hash16 = blake3_files_bin16(&weight_paths)?;
        let tokenizer_hash16 = blake3_file_bin16(&tok_path)?;
        if !cli.force_mismatch {
            if model_hash16 != header.model_hash16 {
                eprintln!("DEBUG: Model hash mismatch! Encoded: {:?}, Decoded: {:?}", model_hash16, header.model_hash16);
                anyhow::bail!("model hash mismatch. Use --force-mismatch to override.");
            }
            if tokenizer_hash16 != header.tokenizer_hash16 {
                eprintln!("DEBUG: Tokenizer hash mismatch! Encoded: {:?}, Decoded: {:?}", tokenizer_hash16, header.tokenizer_hash16);
                anyhow::bail!("tokenizer hash mismatch. Use --force-mismatch to override.");
            }
        }
        let tokenizer = HfTokenizer::from_file(tok_path.to_str().unwrap()).map_err(|e| anyhow::anyhow!("failed loading tokenizer: {e}"))?;
        let mut session = SmolLmSession::load(&weight_paths, &config_path, device, vocab_size)?;
        let mut payload = Vec::new(); rdr.read_to_end(&mut payload)?; let mut acd = ArithmeticDecoder::new(&payload[..])?; let mut logits_t = session.step_logits_tensor(header.bos_token_id)?;
        let mut out_tokens: Vec<u32> = Vec::with_capacity(header.token_count as usize + 1); out_tokens.push(header.bos_token_id);
        let bar = ProgressBar::new(header.token_count); bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());
        let mut logits_vec = logits_t.to_vec1::<f32>()?; let mut pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min());
        let effective_context = (header.context_window as usize).saturating_sub(1).min(511);
        let mut tokens_since_reprime = 0usize;
        for i in 0..header.token_count {
            if session.index_pos() >= effective_context && tokens_since_reprime >= header.reprime_interval as usize {
                let end = out_tokens.len(); let start = end.saturating_sub(effective_context); let history = &out_tokens[start..end]; logits_t = session.reprime_with_history_and_get_last_logits_tensor(history)?; tokens_since_reprime = 0; logits_vec = logits_t.to_vec1::<f32>()?; pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min());
            }
            let sym = acd.decode_symbol(&pdf)? as u32; out_tokens.push(sym); logits_t = session.step_logits_tensor(sym)?; tokens_since_reprime += 1; logits_vec = logits_t.to_vec1::<f32>()?; pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min()); if i % 1024 == 0 { bar.set_position(i); }
        }
        bar.finish_and_clear(); let detok = tokenizer.decode(&out_tokens[1..], true).map_err(|e| anyhow::anyhow!("tokenizer.decode failed: {e}"))?; fs::write(output, detok.as_bytes())?; Ok(())
    } else if backend=="rwkv7" {
        let model_path = cli.rwkv_model.clone();
        let config_path = cli.rwkv_config.clone();
        let tok_path = cli.rwkv_tokenizer.clone();
        let model_hash16 = blake3_file_bin16(&model_path)?;
        let tokenizer_hash16 = blake3_file_bin16(&tok_path)?;
    if !cli.force_mismatch {
        if model_hash16 != header.model_hash16 {
            eprintln!("DEBUG: Model hash mismatch! Encoded: {:?}, Decoded: {:?}", model_hash16, header.model_hash16);
            anyhow::bail!("model hash mismatch. Use --force-mismatch to override.");
        }
        if tokenizer_hash16 != header.tokenizer_hash16 {
            eprintln!("DEBUG: Tokenizer hash mismatch! Encoded: {:?}, Decoded: {:?}", tokenizer_hash16, header.tokenizer_hash16);
            anyhow::bail!("tokenizer hash mismatch. Use --force-mismatch to override.");
        }
    }
        let tokenizer = candlerwkv7::models::rwkv7::Tokenizer::new(&tok_path)?;
        let mut session = Rwkv7Session::load(&model_path, &config_path, device)?;
        let mut payload = Vec::new();
        rdr.read_to_end(&mut payload)?;
        let mut acd = ArithmeticDecoder::new(&payload[..])?;
        let mut logits_t = session.step_logits_tensor(header.bos_token_id)?;
        let mut out_syms: Vec<u32> = Vec::with_capacity(header.token_count as usize + 1);
        out_syms.push(header.bos_token_id);
        let bar = ProgressBar::new(header.token_count);
        bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());

        // RWKV7 decoding with proper literal handling
        for i in 0..header.token_count {
            let logits_vec = logits_t.to_vec1::<f32>()?;
            let pdf = combined_pdf_with_literals(&logits_vec, session.vocab_size());
            let sym = acd.decode_symbol(&pdf)? as u32;
            out_syms.push(sym);

            // Only step the model if it's a valid token (not literal)
            if (sym as usize) < session.vocab_size() {
                logits_t = session.step_logits_tensor(sym)?;
            }

            if i % 256 == 0 { bar.set_position(i); }
        }
        bar.finish_and_clear();

        // Convert combined symbol stream back to bytes
        let detok_bytes = rwkv_detok_with_literals(&tokenizer, &out_syms[1..], session.vocab_size());
        fs::write(output, &detok_bytes)?;
        Ok(())
    } else {
        anyhow::bail!("unknown backend {}", backend);
    }
}



