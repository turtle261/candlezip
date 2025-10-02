use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::io::{self as stdio, BufRead};
use anyhow::{Result, bail, Context};
use blake3::Hasher;
use clap::{Parser, Subcommand, ValueHint};
use hf_hub::api::sync::Api;
use indicatif::{ProgressBar, ProgressStyle};
use serde_json::json;
use tokenizers::Tokenizer as HfTokenizer;
use chrono::Utc;
use csv;

mod models;
use models::{LanguageModelSession, SmolLmSession, Rwkv7Session};

#[derive(Parser, Debug, Clone)]
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
    #[arg(long, default_value_t = 512)]
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
    #[arg(long, default_value_t = 360)]
    scan_agent_timeout: u64,

    /// Gate only if relative savings >= this percent (0-100)
    #[arg(long, default_value_t = 0.0)]
    scan_gate_threshold_pct: f64,

    /// Gate only if absolute savings >= this many bits (set <=0 to disable)
    #[arg(long, default_value_t = -1.0)]
    scan_gate_threshold_abs_bits: f64,

    /// Run per-server attribution and log tool_proof.csv
    #[arg(long, default_value_t = false)]
    scan_attribution: bool,

    /// Enable Agentic Conditioning (deterministic agent priming at chunk boundaries)
    #[arg(long, default_value_t = false)]
    agent: bool,

    /// Agent entry script to execute. Default uses CrewAI Python agent; pass deterministic_mock_agent_cli.py for fully deterministic runs.
    #[arg(long, value_hint = ValueHint::FilePath, default_value = "agent/agent_v2.py")]
    scan_agent_script: PathBuf,

    /// Allow decode even if model/tokenizer BLAKE3 do not match the container
    #[arg(long, default_value_t = false)]
    force_mismatch: bool,

    /// For self-test: reuse cached agent outputs from previous encode run for deterministic decode verification
    #[arg(long, default_value_t = false)]
    reuse: bool,

    #[command(subcommand)]
    command: Commands,

    /// Reuse gate decisions and agent outputs from a previous scan directory (for iterative testing)
    #[arg(long, value_hint = ValueHint::DirPath)]
    reuse_scan_dir: Option<PathBuf>,
    
    /// Policy for tool selection: aligned (default), randomized, leave-one-out:<tool_id>, always-on, react-unpriced, retrieval-only
    #[arg(long, default_value = "aligned")]
    policy: String,
}

#[derive(Subcommand, Debug, Clone)]
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

    // * Enable research-grade watchdog logging for self-test runs (deterministic, side-effect free)
    let ts = Utc::now().format("%Y%m%d_%H%M%S");
    let run_dir = Path::new("scan_output").join(format!(
        "{}_selftest_{}_{}",
        input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("input"),
        cli.backend.to_lowercase(),
        ts
    ));
    fs::create_dir_all(&run_dir).ok();
    std::env::set_var("CANDLEZIP_WATCHDOG", "1");
    std::env::set_var("CANDLEZIP_WATCHDOG_DIR", &run_dir);

    let t0 = std::time::Instant::now();
    encode_file(cli, input, &tmp_out)?;
    let t1 = std::time::Instant::now();

    // Decode with watchdog comparison against the encode trace
    // Enable --reuse for deterministic decode during self-test
    let mut decode_cli = cli.clone();
    decode_cli.reuse = true;
    let decode_res = decode_file(&decode_cli, &tmp_out, &tmp_dec);
    let t2 = std::time::Instant::now();

    let src = fs::read(input)?;
    let enc = fs::read(&tmp_out)?;
    let dec = fs::read(&tmp_dec)?;

    let bits_per_byte = (8.0 * enc.len() as f64) / (src.len() as f64);
    println!("Compression   : {:.2?}", t1 - t0);
    println!("Decompression : {:.2?}", t2 - t1);
    println!("Bits per byte : {:.3}", bits_per_byte);

    if decode_res.is_ok() && src == dec {
        println!("Roundtrip OK. Encoded: {} bytes, Decoded: {} bytes", enc.len(), dec.len());
        Ok(())
    } else {
        eprintln!("[watchdog] Roundtrip mismatch detected. Collecting diagnostics...");
        // Print tails of encode/decode watchdog streams and mismatch report if present
        let enc_path = run_dir.join("watchdog_encode_steps.jsonl");
        let dec_path = run_dir.join("watchdog_decode_steps.jsonl");
        let mm_path = run_dir.join("watchdog_mismatch.json");
        let tail = |p: &Path| -> Vec<String> {
            if !p.exists() { return vec![format!("<missing: {}>", p.display())]; }
            let Ok(text) = fs::read_to_string(p) else { return vec![format!("<unreadable: {}>", p.display())]; };
            let mut lines: Vec<&str> = text.lines().collect();
            let n = lines.len();
            if n > 20 { lines = lines[n-20..].to_vec(); }
            lines.into_iter().map(|s| s.to_string()).collect()
        };
        eprintln!("[watchdog] encode tail:");
        for l in tail(&enc_path) { eprintln!("{}", l); }
        eprintln!("[watchdog] decode tail:");
        for l in tail(&dec_path) { eprintln!("{}", l); }
        if mm_path.exists() {
            if let Ok(mm) = fs::read_to_string(&mm_path) { eprintln!("[watchdog] mismatch: {}", mm); }
        }
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
fn ac_p_min() -> f64 {
    let base = AC_BASE as f64;
    2.0 * base.powi(-(AC_PRECISION as i32 - 2))
}

// * Reserved flags bit layout (VERSION=2):
// * bit 0: agent conditioning used (1=yes)
// * bits 1..15: reserved (0)
// * bits 16..31: agent chunk size (tokens), 16-bit unsigned
const FLAG_AGENT_USED: u32 = 1 << 0;
const FLAG_AGENT_MOCK: u32 = 1 << 1;
const FLAG_AGENT_GATES: u32 = 1 << 2; // gating bits present after header
fn flags_pack(agent_used: bool, agent_mock: bool, gates_present: bool, agent_chunk: usize) -> u32 {
    let mut f = 0u32;
    if agent_used { f |= FLAG_AGENT_USED; }
    if agent_mock { f |= FLAG_AGENT_MOCK; }
    if gates_present { f |= FLAG_AGENT_GATES; }
    let chunk16 = (agent_chunk as u32) & 0xFFFF;
    f |= chunk16 << 16;
    f
}
fn flags_unpack_agent_used(flags: u32) -> bool { (flags & FLAG_AGENT_USED) != 0 }
fn flags_unpack_agent_mock(flags: u32) -> bool { (flags & FLAG_AGENT_MOCK) != 0 }
fn flags_unpack_agent_gates(flags: u32) -> bool { (flags & FLAG_AGENT_GATES) != 0 }
fn flags_unpack_agent_chunk(flags: u32) -> usize { ((flags >> 16) & 0xFFFF) as usize }

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

    fn finish(mut self) -> anyhow::Result<W> {
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
        Ok(self.out)
    }

    fn bytes_written(&self) -> u64 { self.bytes_out }
    #[allow(dead_code)]
    fn into_inner(self) -> W { self.out }
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
    // no-op
    while v >= 0x80 {
        w.write_all(&[((v as u8) & 0x7F) | 0x80])?;
        v >>= 7;
    }
    w.write_all(&[v as u8])?;
    Ok(())
}

fn read_var_u64<R: std::io::Read>(r: &mut R) -> anyhow::Result<u64> {
    // no-op
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
// Agent gating metadata (binary)
// ----------------------------------

const AGTB_MAGIC: [u8; 4] = *b"AGTB"; // marker for gating bits section
const AGT2_MAGIC: [u8; 4] = *b"AGT2"; // structured gating section (v2)

#[derive(Clone, Copy, Debug, Default)]
struct GateRecordV2 { gate: u8, candidate_id: u8, budget_id: u8 }

fn write_agent_gates_v2<W: std::io::Write>(w: &mut W, records: &[GateRecordV2]) -> anyhow::Result<()> {
    // no-op
    w.write_all(&AGT2_MAGIC)?;
    write_var_u64(w, records.len() as u64)?;
    for rec in records {
        let mut byte: u8 = 0;
        byte |= (rec.gate & 1) << 0;
        byte |= (rec.candidate_id & 0b11) << 1;
        byte |= (rec.budget_id & 0b11) << 3;
        w.write_all(&[byte])?;
    }
    Ok(())
}

fn get_gate_bit(gates_bytes: &[u8], idx: usize) -> u8 {
    let byte = idx / 8;
    let bit = idx % 8;
    if byte >= gates_bytes.len() { return 0; }
    ((gates_bytes[byte] >> bit) & 1) as u8
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
// Watchdog helpers (research-grade, deterministic)
// ----------------------------------

fn hex16(bytes: &[u8; 16]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = [0u8; 32];
    for (i, b) in bytes.iter().enumerate() {
        out[2 * i] = HEX[(b >> 4) as usize];
        out[2 * i + 1] = HEX[(b & 0x0F) as usize];
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn blake3_f32_bin16(values: &[f32]) -> [u8; 16] {
    let mut hasher = Hasher::new();
    for v in values { hasher.update(&v.to_le_bytes()); }
    let mut out = [0u8; 16];
    out.copy_from_slice(hasher.finalize().as_bytes()[..16].as_ref());
    out
}

// ============================================================================
// SIMDL v1.1: Document Index and Pricing Functions
// ============================================================================

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct DocumentEntry {
    doc_id: usize,
    domain: String,
    file_name: String,
    size_bytes: usize,
}

#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
struct DocumentIndex {
    documents: Vec<DocumentEntry>,
    max_span_len_bytes: usize,
}

impl DocumentIndex {
    fn new() -> Self {
        Self {
            documents: Vec::new(),
            max_span_len_bytes: 1024, // Default max span length
        }
    }
    
    fn add_document(&mut self, domain: &str, file_name: &str, size_bytes: usize) -> usize {
        let doc_id = self.documents.len();
        self.documents.push(DocumentEntry {
            doc_id,
            domain: domain.to_string(),
            file_name: file_name.to_string(),
            size_bytes,
        });
        doc_id
    }
    
    fn get_document(&self, doc_id: usize) -> Option<&DocumentEntry> {
        self.documents.get(doc_id)
    }
    
    fn save_to_path(&self, path: &Path) -> Result<()> {
        let json_str = serde_json::to_string_pretty(self)?;
        fs::write(path, json_str)?;
        Ok(())
    }
    
    fn load_from_path(path: &Path) -> Result<Self> {
        if path.exists() {
            let json_str = fs::read_to_string(path)?;
            let index: DocumentIndex = serde_json::from_str(&json_str)?;
            Ok(index)
        } else {
            Ok(DocumentIndex::new())
        }
    }
}

/// Compute transcript pricing using zstd compression
fn compute_transcript_price_bits(transcript_text: &str) -> Result<u64> {
    let transcript_bytes = transcript_text.as_bytes();
    let compressed = zstd::bulk::compress(transcript_bytes, 19)?;
    Ok((compressed.len() * 8) as u64)
}

/// Compute pointer pricing based on document index
fn compute_pointer_price_bits(doc_index: &DocumentIndex, doc_id: usize, _start_offset: usize, _span_len: usize) -> Result<u64> {
    let doc_entry = doc_index.get_document(doc_id)
        .ok_or_else(|| anyhow::anyhow!("Document ID {} not found in index", doc_id))?;
    
    let n_docs = doc_index.documents.len();
    let doc_size_bytes = doc_entry.size_bytes;
    let max_span_bytes = doc_index.max_span_len_bytes;
    
    // Calculate required bits for each component
    let doc_id_bits = if n_docs <= 1 { 1 } else { (n_docs as f64).log2().ceil() as u64 };
    let start_bits = if doc_size_bytes <= 1 { 1 } else { (doc_size_bytes as f64).log2().ceil() as u64 };
    let span_bits = if max_span_bytes <= 1 { 1 } else { (max_span_bytes as f64).log2().ceil() as u64 };
    
    Ok(doc_id_bits + start_bits + span_bits)
}

/// Extract domain and identifiers from file path for SIMDL grouping
fn extract_simdl_identifiers(input_path: &Path, agent_script: &Path, policy: &str) -> (String, String, String, String, String) {
    let domain = input_path.parent()
        .and_then(|p| p.file_name())
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
        
    let agent_id = agent_script.file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
        
    let toolset_id = format!("policy_{}", policy);
    
    let run_id = Utc::now().format("%Y%m%d_%H%M%S").to_string();
    
    let chunk_id_prefix = input_path.file_stem()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();
        
    (domain, agent_id, toolset_id, run_id, chunk_id_prefix)
}

fn watchdog_dir() -> Option<PathBuf> {
    match std::env::var("CANDLEZIP_WATCHDOG") {
        Ok(v) if v == "1" => std::env::var("CANDLEZIP_WATCHDOG_DIR").ok().map(PathBuf::from),
        _ => None,
    }
}

fn watchdog_write_jsonl(dir: &Path, file: &str, v: &serde_json::Value) {
    let path = dir.join(file);
    if let Ok(mut f) = File::options().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{}", v);
    }
}

fn watchdog_try_write_json(dir: &Path, file: &str, v: &serde_json::Value) {
    let path = dir.join(file);
    // Best-effort write; overwrite allowed to keep last mismatch
    if let Ok(mut f) = File::create(&path) {
        let _ = writeln!(f, "{}", v);
    }
}

#[derive(Clone, Debug)]
struct EncodeStepRef {
    sym: u32,
    #[allow(dead_code)]
    pdf_digest: Option<String>,
}

fn watchdog_load_encode_trace(dir: &Path) -> Vec<Option<EncodeStepRef>> {
    let path = dir.join("watchdog_encode_steps.jsonl");
    let Ok(text) = fs::read_to_string(&path) else { return Vec::new() };
    let mut out: Vec<Option<EncodeStepRef>> = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            let phase = v.get("phase").and_then(|x| x.as_str()).unwrap_or("");
            if phase == "encode_step" {
                if let (Some(i), Some(sym)) = (v.get("i").and_then(|x| x.as_u64()), v.get("sym").and_then(|x| x.as_u64())) {
                    let idx = i as usize;
                    if out.len() <= idx { out.resize(idx + 1, None); }
                    let pdf_digest = v.get("pdf_digest").and_then(|x| x.as_str()).map(|s| s.to_string());
                    out[idx] = Some(EncodeStepRef { sym: sym as u32, pdf_digest });
                }
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
struct AgentResultCache {
    #[allow(dead_code)]
    chunk_index: usize,
    agent_text: String,
    agent_calls: u32,
}

fn cache_agent_result(dir: &Path, chunk_index: usize, agent_text: &str, agent_calls: u32) {
    let rec = json!({
        "chunk_index": chunk_index,
        "agent_text": agent_text,
        "agent_calls": agent_calls,
    });
    watchdog_write_jsonl(dir, "agent_cache.jsonl", &rec);
}

fn load_cached_agent_results(dir: &Path) -> std::collections::HashMap<usize, AgentResultCache> {
    let path = dir.join("agent_cache.jsonl");
    let Ok(text) = fs::read_to_string(&path) else { return std::collections::HashMap::new() };
    let mut cache = std::collections::HashMap::new();
    for line in text.lines() {
        if line.trim().is_empty() { continue; }
        if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
            if let (Some(idx), Some(text), Some(calls)) = (
                v.get("chunk_index").and_then(|x| x.as_u64()),
                v.get("agent_text").and_then(|x| x.as_str()),
                v.get("agent_calls").and_then(|x| x.as_u64())
            ) {
                cache.insert(idx as usize, AgentResultCache {
                    chunk_index: idx as usize,
                    agent_text: text.to_string(),
                    agent_calls: calls as u32,
                });
            }
        }
    }
    cache
}

fn load_gate_decisions_from_csv(dir: &Path) -> std::collections::HashMap<usize, (u8, u8, u8)> {
    // Returns map of chunk_index -> (gate, candidate_id, budget_id)
    let path = dir.join("proof.csv");
    let Ok(text) = fs::read_to_string(&path) else { return std::collections::HashMap::new() };
    let mut gates = std::collections::HashMap::new();
    for (li, line) in text.lines().enumerate() {
        if li == 0 || line.trim().is_empty() { continue; } // Skip header
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() >= 13 {
            if let (Ok(chunk_idx), Ok(gate), Ok(cand_id), Ok(bud_id)) = (
                parts[1].trim().parse::<usize>(),
                parts[11].trim().parse::<u8>(),
                parts[12].trim().parse::<u8>(),
                parts[13].trim().parse::<u8>(),
            ) {
                gates.insert(chunk_idx, (gate, cand_id, bud_id));
            }
        }
    }
    gates
}

// ----------------------------------
// Scan helpers
// ----------------------------------

fn pick_python(exe: &str) -> String { if exe != "auto" { return exe.to_string(); } "python".to_string() }

fn find_agent_rs_binary() -> anyhow::Result<PathBuf> {
    // Prefer an explicit absolute path on Windows per user preference
    #[cfg(target_os = "windows")]
    {
        let preferred = PathBuf::from(r"C:\Users\Noah\Documents\business\resumeportfolio\candlezip\agent-rs\target\release\agent-rs.exe");
        if preferred.exists() { return Ok(preferred); }
    }
    // Workspace-relative common paths
    let candidates = [
        PathBuf::from("agent-rs/target/release/agent-rs"),
        PathBuf::from("agent-rs/target/release/agent-rs.exe"),
        PathBuf::from("target/release/agent-rs"),
        PathBuf::from("target/release/agent-rs.exe"),
    ];
    for c in candidates.iter() { if c.exists() { return Ok(c.clone()); } }
    // Search PATH
    if let Some(paths) = std::env::var_os("PATH") {
        for p in std::env::split_paths(&paths) {
            let exe = p.join("agent-rs"); if exe.exists() { return Ok(exe); }
            let exe_win = p.join("agent-rs.exe"); if exe_win.exists() { return Ok(exe_win); }
        }
    }
    anyhow::bail!("agent-rs binary not found. Build it first (cargo build --release) or run build.ps1")
}

fn select_agent_runner(agent_script: &Path, python_exe: &str) -> anyhow::Result<(bool, PathBuf, String)> {
    // Returns: (is_python, path_or_script, python_exe)
    if agent_script.extension().and_then(|s| s.to_str()).unwrap_or("") == "py" {
        if agent_script.exists() { return Ok((true, agent_script.to_path_buf(), pick_python(python_exe))); }
    }
    // Fall back to rig binary
    let bin = find_agent_rs_binary()?;
    Ok((false, bin, String::new()))
}

fn read_agent_memory(scan_dir: &Path) -> String {
    let mem_file = scan_dir.join("agent_memory").join("memory.txt");
    if let Ok(bytes) = fs::read(&mem_file) {
        if let Ok(text) = String::from_utf8(bytes) { return text; }
    }
    String::new()
}

fn append_agent_memory(scan_dir: &Path, chunk_index: usize, agent_text: &str) {
    let mem_dir = scan_dir.join("agent_memory"); let _ = fs::create_dir_all(&mem_dir);
    let mem_file = mem_dir.join("memory.txt");
    let rec = format!("\n# chunk:{}\n{}\n", chunk_index, agent_text.trim());
    if let Ok(mut f) = File::options().create(true).append(true).open(mem_file) { let _ = f.write_all(rec.as_bytes()); }
}

// * Scratchpad aggregates prior prefix text and accepted gated hints for agent reflection (ephemeral under scan_dir)
fn scratchpad_path(scan_dir: &Path) -> PathBuf { scan_dir.join("agent_memory").join("scratchpad.txt") }

fn append_scratchpad(scan_dir: &Path, section: &str, content: &str) {
    let path = scratchpad_path(scan_dir); let _ = fs::create_dir_all(scan_dir.join("agent_memory"));
    let mut rec = String::new();
    rec.push_str("\n==== "); rec.push_str(section); rec.push_str(" ====" ); rec.push('\n');
    rec.push_str(content.trim()); rec.push('\n');
    let _ = fs::OpenOptions::new().create(true).append(true).open(&path).and_then(|mut f| f.write_all(rec.as_bytes()));
}

// * Learning data: write directly from Rust after gating decision for the SAME chunk.
//   This avoids off-by-one and ensures consistency with proof.csv.
fn write_learning_entry(
    scan_dir: &Path,
    input_file: &Path,
    chunk_index: usize,
    gate: u8,
    bits_saved: f64,
    baseline_bits: f64,
    candidate_id: usize,
    budget_id: usize,
    agent_text: &str,
) -> anyhow::Result<()> {
    // no-op
    // Mirror Python naming: candlezip_{file_stem}_{md5(abs_path)[:8]}_learning.jsonl
    let memory_dir = scan_dir.join("agent_memory");
    fs::create_dir_all(&memory_dir)?;
    let abs_str = input_file.canonicalize().unwrap_or(input_file.to_path_buf()).to_string_lossy().to_string();
    let md5_8 = {
        let digest = md5::compute(abs_str.as_bytes());
        format!("{:x}", digest)[..8].to_string()
    };
    let file_stem = input_file.file_stem().and_then(|s| s.to_str()).unwrap_or("unknown");
    let learning_file = memory_dir.join(format!("candlezip_{}_{}_learning.jsonl", file_stem, md5_8));

    let success = gate == 1;
    let summary = if success {
        format!(
            "COMPRESSION SUCCESS (Chunk {}): Your prediction strategy succeeded! You saved {:.2} bits (baseline: {:.2} bits). The successful approach used candidate {} with budget {}. Key success factors: Your output provided precise contextual information that reduced prediction uncertainty. Remember this pattern and strategy for similar text content in future chunks.",
            chunk_index, bits_saved, baseline_bits, candidate_id, budget_id
        )
    } else {
        format!(
            "COMPRESSION ATTEMPT (Chunk {}): Your prediction did not improve compression (saved {:.2} bits from baseline {:.2}). The attempted approach used candidate {} with budget {}. Learning opportunity: Consider different prediction strategies, more specific content, or alternative tool usage patterns for this type of text content.",
            chunk_index, bits_saved, baseline_bits, candidate_id, budget_id
        )
    };

    let learning_entry = json!({
        "type": "compression_learning",
        "chunk_index": chunk_index,
        "file_path": input_file.to_string_lossy(),
        "gate_result": gate,
        "bits_saved": bits_saved,
        "baseline_bits": baseline_bits,
        "candidate_id": candidate_id,
        "budget_id": budget_id,
        "agent_output": agent_text,
        "success": success,
        "timestamp": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs_f64(),
        "summary": summary
    });

    if let Ok(mut f) = fs::OpenOptions::new().create(true).append(true).open(&learning_file) {
        writeln!(f, "{}", serde_json::to_string(&learning_entry)?)?;
        eprintln!("[Learning] Recorded chunk {} -> gate={}, bits_saved={:.2}", chunk_index, gate, bits_saved);
    }
    Ok(())
}

// * Agent output hygiene & caching for cross-run consistency
fn sanitize_agent_text(raw: &str) -> String {
    // * Removes code fences, tool chatter, and reasoning labels; returns plain text
    let s0 = raw.replace("\r", "");
    // Strip code fences by reconstructing without fenced blocks
    let mut s = String::new();
    let chars: Vec<char> = s0.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if i + 3 <= chars.len() && chars[i] == '`' && chars[i+1] == '`' && chars[i+2] == '`' {
            i += 3;
            // Skip to next ```
            while i + 2 < chars.len() {
                if chars[i] == '`' && chars[i+1] == '`' && chars[i+2] == '`' {
                    i += 3;
                    break;
                }
                i += 1;
            }
        } else {
            s.push(chars[i]);
            i += 1;
        }
    }
    // Drop lines that are clearly meta/non-content
    let mut out_lines = Vec::new();
    for line in s.lines() {
        let l = line.trim();
        if l.is_empty() { continue; }
        if l.starts_with("Thought:") || l.starts_with("Reasoning Plan:") || l.starts_with("Crew Execution") || l.starts_with("Task:") || l.starts_with("Final Answer JSON:") { continue; }
        if l.starts_with("AGENT_RESULT_JSON:") { continue; }
        // Filter out "noise" text and other artifacts
        if l == "noise" || l.starts_with("noise") || l.contains("Event loop is closed") || l.contains("Tool Usage Failed") || l.contains("RuntimeWarning") { continue; }
        // Filter out empty or very short lines that are likely artifacts
        if l.len() < 10 && !l.chars().any(|c| c.is_alphabetic()) { continue; }
        out_lines.push(l);
    }
    let out = out_lines.join("\n");
    out.trim().to_string()
}

fn agent_cache_dir() -> PathBuf { Path::new("scan_output").join("agent_cache") }

fn agent_cache_key(input: &Path, chunk_index: usize, prefix_text: &str) -> String {
    // * Context-aware key: input path + chunk_index + full prefix hash to prevent cross-chunk contamination
    let mut hasher = blake3::Hasher::new();
    hasher.update(input.to_string_lossy().as_bytes());
    hasher.update(&chunk_index.to_le_bytes());
    // Use the entire prefix to ensure uniqueness across chunks
    hasher.update(prefix_text.as_bytes());
    hex16(&hasher.finalize().as_bytes()[..16].try_into().unwrap())
}

fn agent_cache_get(input: &Path, chunk_index: usize, prefix_text: &str) -> Option<String> {
    let dir = agent_cache_dir(); let _ = fs::create_dir_all(&dir);
    let key = agent_cache_key(input, chunk_index, prefix_text);
    let path = dir.join(format!("{}.txt", key));
    if let Ok(bytes) = fs::read(&path) { if let Ok(text) = String::from_utf8(bytes) { return Some(text); } }
    None
}

fn agent_cache_put(input: &Path, chunk_index: usize, prefix_text: &str, agent_text: &str) {
    let dir = agent_cache_dir(); let _ = fs::create_dir_all(&dir);
    let key = agent_cache_key(input, chunk_index, prefix_text);
    let path = dir.join(format!("{}.txt", key));
    let _ = fs::write(path, agent_text.as_bytes());
}


fn tail_str(s: &str, n_chars: usize) -> String {
    let total = s.chars().count();
    if total <= n_chars { return s.to_string(); }
    let start_idx = s.char_indices().nth(total - n_chars).map(|(i, _)| i).unwrap_or(0);
    s[start_idx..].to_string()
}

fn build_agent_task(prefix_text: &str, prior_memory: &str) -> String {
    let prefix_trunc_s = tail_str(prefix_text, 8000);
    let memory_trunc_s = tail_str(prior_memory, 8000);
    format!(
        "You MUST use MCP tools aggressively to find the exact immediate continuation that follows this prefix.\n- Search for the source document or similar content to extract the next 100-200 words verbatim.\n- Use Wikipedia, search tools, and any available knowledge sources to locate the full context.\n- If you find the exact source, copy the immediate continuation word-for-word.\n- If no exact source is found, use search and knowledge tools to predict the most likely next text based on context.\n- Prioritize accuracy and relevance over creativity.\n- Output MUST be plain text continuation only (no markdown, no analysis, no commentary).\n- Avoid any formatting, lists, headings, or meta-text.\n- Focus on the immediate next words/sentences that naturally follow the prefix.\n\nIf ALL tools fail:\n- Generate a continuation based on the current prefix context only.\n- Do NOT reuse previous chunk content - analyze the current prefix and predict what would naturally follow.\n- Make the continuation as specific to the current text as possible.\n- Avoid generic text that could apply to any context.\n\nPrior memory (from earlier chunks):\n{}\n\nCurrent document prefix (UTF-8 text):\n{}\n\nOutput: continuation (plain text only).",
        &memory_trunc_s,
        &prefix_trunc_s,
    )
}

fn run_agent_for_chunk(
    _input_path: &Path,
    full_data: &[u8],
    prefix_end_byte: usize,
    chunk_index: usize,
    agent_script: &Path,
    python_exe: &str,
    mcp_config: &Path,
    max_steps: usize,
    verbose: bool,
    scan_dir: &Path,
    agent_timeout_secs: u64,
) -> anyhow::Result<(String, u64, u32)> {
    let prefix = &full_data[..prefix_end_byte.min(full_data.len())];
    let prefix_text = String::from_utf8_lossy(prefix);
    let prior_memory = read_agent_memory(scan_dir);
    let task = build_agent_task(&prefix_text, &prior_memory);

    let (use_python, runner_path, py_exe) = select_agent_runner(agent_script, python_exe)?;
    let log_path = scan_dir.join(format!("agent_chunk_{}.log", chunk_index));

    // Merge env with agent/.env if present
    let mut env_kv: Vec<(String, String)> = std::env::vars().collect();
    let agent_env_path = Path::new("agent").join(".env");
    if let Ok(bytes) = fs::read(&agent_env_path) {
        if let Ok(text) = String::from_utf8(bytes) {
            for line in text.lines() {
                let s = line.trim(); if s.is_empty() || s.starts_with('#') { continue; }
                if let Some(eq) = s.find('=') { let (k, v) = s.split_at(eq); let v = &v[1..]; env_kv.push((k.trim().to_string(), v.trim().to_string())); }
            }
        }
    }

    let mut cmd = if use_python {
        let mut c = Command::new(&py_exe);
        c.arg(&runner_path)
            .arg("--task").arg(&task)
            .arg("--mcp-config").arg(mcp_config)
            .arg("--max-steps").arg(max_steps.to_string());
        // Note: scratchpad removed in favor of CrewAI memory integration
        c
    } else {
        let mut c = Command::new(&runner_path);
        c.arg("--task").arg(&task)
        .arg("--mcp-config").arg(mcp_config)
        .arg("--max-steps").arg(max_steps.to_string())
        .arg("--timeout").arg((agent_timeout_secs.saturating_mul(1000)).to_string())
            .arg("--temperature").arg("0.0");
        c
    };
    let mut child = cmd
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
    // Try cache first if configured
    let prefix_text_for_cache = String::from_utf8_lossy(&full_data[..prefix_end_byte.min(full_data.len())]).to_string();
    let use_cache = std::env::var("CANDLEZIP_AGENT_REUSE").map(|v| v == "1" || v.eq_ignore_ascii_case("true")).unwrap_or(false);
    if use_cache {
        if let Some(cached) = agent_cache_get(_input_path, chunk_index, &prefix_text_for_cache) {
            // Don't append to agent memory here - will be handled after gating decision
            return Ok((cached, duration_ms, calls));
        }
    }
    // Deterministic fallback if agent produced no text
    if agent_text.trim().is_empty() {
        // Fallback: last 1000 chars of prefix text, normalized
        let prefix = String::from_utf8_lossy(&full_data[..prefix_end_byte.min(full_data.len())]);
        let mut s = prefix.chars().rev().take(1000).collect::<String>();
        s = s.chars().rev().collect();
        let normalized: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
        let out = sanitize_agent_text(&normalized);
        // Don't cache or append to memory here - will be handled after gating decision
        return Ok((out, duration_ms, calls));
    }
    let cleaned = sanitize_agent_text(&agent_text);
    // Don't cache or append to memory here - will be handled after gating decision
    Ok((cleaned, duration_ms, calls))
}

fn run_agent_for_prefix_text(
    prefix_text: &str,
    chunk_index: usize,
    agent_script: &Path,
    python_exe: &str,
    mcp_config: &Path,
    max_steps: usize,
    verbose: bool,
    scan_dir: &Path,
    agent_timeout_secs: u64,
) -> anyhow::Result<(String, u64, u32)> {
    let prior_memory = read_agent_memory(scan_dir);
    let task = build_agent_task(prefix_text, &prior_memory);

    let (use_python, runner_path, py_exe) = select_agent_runner(agent_script, python_exe)?;
    let log_path = scan_dir.join(format!("agent_chunk_{}.log", chunk_index));

    // Merge env with agent/.env if present
    let mut env_kv: Vec<(String, String)> = std::env::vars().collect();
    let agent_env_path = Path::new("agent").join(".env");
    if let Ok(bytes) = fs::read(&agent_env_path) {
        if let Ok(text) = String::from_utf8(bytes) {
            for line in text.lines() {
                let s = line.trim(); if s.is_empty() || s.starts_with('#') { continue; }
                if let Some(eq) = s.find('=') { let (k, v) = s.split_at(eq); let v = &v[1..]; env_kv.push((k.trim().to_string(), v.trim().to_string())); }
            }
        }
    }

    let mut cmd = if use_python {
        let mut c = Command::new(&py_exe);
        c.arg(&runner_path)
            .arg("--task").arg(&task)
            .arg("--mcp-config").arg(mcp_config)
            .arg("--max-steps").arg(max_steps.to_string());
        // Note: scratchpad removed in favor of CrewAI memory integration
        c
    } else {
        let mut c = Command::new(&runner_path);
        c.arg("--task").arg(&task)
        .arg("--mcp-config").arg(mcp_config)
        .arg("--max-steps").arg(max_steps.to_string())
        .arg("--timeout").arg((agent_timeout_secs.saturating_mul(1000)).to_string())
            .arg("--temperature").arg("0.0");
        c
    };
    let mut child = cmd
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
    if child.try_wait()?.is_none() { let _ = child.kill(); eprintln!("[agent] agent timed out after {:?}; killed.", timeout); }
    let (text_out, dur_out, calls_out) = t_out.join().unwrap_or((String::new(), 0, 0));
    let calls_err = t_err.join().unwrap_or(0);
    if !text_out.is_empty() { agent_text = text_out; }
    if dur_out > 0 { duration_ms = dur_out; }
    calls = calls.saturating_add(calls_out).saturating_add(calls_err);
    // Do not synthesize or duplicate; return possibly empty if agent yielded nothing
    // Don't append to agent memory here - will be handled after gating decision in decode path
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
    let mut hint_slice: &[u32] = &[];
    if let Some(h) = hint { hint_slice = h; }
    // Use caller-provided hint as-is; budget should be enforced by caller (aligned with encode budgets)
    let hint_budget = hint_slice.len().min(max_ctx);
    let remaining_budget = max_ctx.saturating_sub(hint_budget);
    let hist_take = remaining_budget.min(history.len());
    if hist_take > 0 {
        let hist_start = history.len() - hist_take;
    prime.extend_from_slice(&history[hist_start..]);
    }
    if hint_budget > 0 { prime.extend_from_slice(&hint_slice[..hint_budget]); }
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

fn cross_entropy_bits_over_span_rwkv(
    session: &mut Rwkv7Session,
    history: &[u32],
    targets: &[u32],
    hint: Option<&[u32]>,
    vocab_size: usize,
) -> anyhow::Result<f64> {
    if targets.is_empty() { return Ok(0.0); }
    let max_ctx = session.max_context_length().saturating_sub(1);
    let mut prime: Vec<u32> = Vec::new();
    // Filter out literals from history for RWKV reprime stability
    let mut filtered_history: Vec<u32> = Vec::with_capacity(history.len());
    for &t in history { if (t as usize) < vocab_size { filtered_history.push(t); } }
    let mut hint_slice: &[u32] = &[];
    if let Some(h) = hint { hint_slice = h; }
    let hint_budget = hint_slice.len().min(max_ctx);
    let remaining_budget = max_ctx.saturating_sub(hint_budget);
    let take = remaining_budget.min(filtered_history.len());
    if take > 0 { prime.extend_from_slice(&filtered_history[filtered_history.len() - take..]); }
    if hint_budget > 0 { prime.extend_from_slice(&hint_slice[..hint_budget]); }
    if prime.is_empty() { anyhow::bail!("reprime called with empty history (rwkv xe)"); }
    let mut logits_t = session.reprime_with_history_and_get_last_logits_tensor(&prime)?;
    let mut bits: f64 = 0.0;
    for &sym in targets.iter() {
        let logits_vec = logits_t.to_vec1::<f32>()?;
        let pdf = combined_pdf_with_literals(&logits_vec, vocab_size);
        let idx = sym as usize;
        let p = if idx < pdf.len() { pdf[idx] } else { ac_p_min() };
        bits += -p.max(1e-300).log2();
        if (sym as usize) < vocab_size {
            logits_t = session.step_logits_tensor(sym)?;
        }
    }
    Ok(bits)
}

// ----------------------------------
// Encode / Decode (unified)
// ----------------------------------

fn encode_file(cli: &Cli, input: &Path, output: &Path) -> anyhow::Result<()> {
    let t0 = std::time::Instant::now();
    // reuse scan directory flag (defined below when scan is initialized)
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
    
    // Initialize SIMDL v1.1 document index and identifiers
    let mut doc_index = DocumentIndex::load_from_path(&Path::new("index_meta.json")).unwrap_or_else(|_| DocumentIndex::new());
    // Deduplicate: if (domain, file_name, size_bytes) exists, reuse its doc_id
    let domain_name = input.parent().and_then(|p| p.file_name()).and_then(|n| n.to_str()).unwrap_or("unknown");
    let file_name_only = input.file_name().and_then(|n| n.to_str()).unwrap_or("unknown").to_string();
    let size_b = orig_len_bytes as usize;
    let mut existing = None;
    for e in &doc_index.documents {
        if e.domain == domain_name && e.file_name == file_name_only && e.size_bytes == size_b {
            existing = Some(e.doc_id);
            break;
        }
    }
    let doc_id = match existing { Some(id) => id, None => doc_index.add_document(&domain_name, &file_name_only, size_b) };
    let (domain, agent_id, toolset_id, run_id, chunk_id_prefix) = extract_simdl_identifiers(input, &cli.scan_agent_script, &cli.policy);

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

    // * Determine agent chunk size (shared with scan if enabled)
    let agent_enabled = cli.agent;
    // Keyword augmentation removed from agent prompt for domain neutrality; tokenizer path remains deterministic.
    let agent_chunk = if backend=="rwkv7" { cli.scan_chunk_size.unwrap_or(2048).max(1) } else { cli.scan_chunk_size.unwrap_or(cli.reprime_interval).max(1) };
    // If user selected deterministic mock agent script, flag it
    let agent_script_path = cli.scan_agent_script.clone();
    let agent_mock = agent_enabled && agent_script_path.file_name().and_then(|s| s.to_str()).unwrap_or("") == "deterministic_mock_agent_cli.py";
    // We will buffer the arithmetic-coded payload in memory to insert gating bits after header deterministically

    // Scan/Agent run setup (shared)
    let scan_enabled = cli.scan; let mut scan_dir = cli.scan_output_dir.clone(); let mut csv_writer_opt: Option<csv::Writer<File>> = None; let mut scan_meta: serde_json::Value = json!({});
    let reuse_scan_mode = cli.reuse_scan_dir.is_some();
    if scan_enabled || agent_enabled || reuse_scan_mode {
        if let Some(reuse_dir) = &cli.reuse_scan_dir {
            // Reuse existing scan directory
            scan_dir = reuse_dir.clone();
            if !scan_dir.exists() { anyhow::bail!("Reuse scan directory does not exist: {}", scan_dir.display()); }
            eprintln!("[reuse] Using existing scan directory: {}", scan_dir.display());
        } else {
            // Check for self-test watchdog directory BEFORE creating any directories to avoid duplication
            if let Ok(dir) = std::env::var("CANDLEZIP_WATCHDOG_DIR") {
                scan_dir = PathBuf::from(dir);
            } else {
                // Per-run dir to prevent cross-run contamination and logs overlap
                let ts = Utc::now().format("%Y%m%d_%H%M%S");
                let run_dir_name = format!("{}_{}_{}", input.file_stem().and_then(|s| s.to_str()).unwrap_or("input"), backend, ts);
                scan_dir = scan_dir.join(run_dir_name);
            }
            fs::create_dir_all(&scan_dir).ok();
        }
        // Create CSV writer only if not reusing (for new runs)
        if !reuse_scan_mode {
            let csv_path=scan_dir.join("proof.csv"); let csv_f=File::create(&csv_path)?; let mut wtr = csv::Writer::from_writer(csv_f);
            // Extend header to include SIMDL v1.1 columns for offline lambda-sweeps and pricing
            wtr.write_record([
                "file","chunk_index","start_token","end_token","agent_text_len","agent_duration_ms",
                "cross_entropy_baseline_bits","cross_entropy_conditioned_bits","bits_saved","percent_saved",
                "agent_calls","gate","candidate_id","budget_id",
                // New SIMDL v1.1 columns
                "gate_bits","price_transcript_bits","price_pointer_bits","tool_id_best",
                "tool_snapshot_id","args_hash","output_hash","domain","agent_id","toolset_id","run_id","chunk_id"
            ]) ?; wtr.flush()?; csv_writer_opt=Some(wtr);
        }
        scan_meta = json!({"input_file": input.to_string_lossy(), "started_at": Utc::now().to_rfc3339(), "backend": backend, "context": cli.context, "reprime_interval": cli.reprime_interval, "agent": {"enabled": agent_enabled, "chunk_size": agent_chunk, "max_steps": cli.scan_max_steps, "max_hint_tokens": cli.scan_max_hint_tokens, "mcp_config": cli.scan_mcp_config.to_string_lossy(), "python": cli.scan_python }, "scan": scan_enabled, "reuse_scan_dir": reuse_scan_mode, "run_dir": scan_dir.to_string_lossy() });
    }

    // AC encode
    let ace_sink: Vec<u8> = Vec::new();
    let mut ace = ArithmeticEncoder::new(ace_sink);
    let mut logits_t = if backend=="smollm" { session_smol.as_mut().unwrap().step_logits_tensor(bos_id)? } else { session_rwkv.as_mut().unwrap().step_logits_tensor(bos_id)? };
    // Watchdog: write initial context
    let wdog = watchdog_dir();
    if let Some(dir) = wdog.as_ref() {
        let rec = json!({
            "phase": "encode_init",
            "backend": backend,
            "vocab": vocab_size,
            "bos": bos_id,
            "context": cli.context,
            "agent": {"enabled": agent_enabled, "chunk": agent_chunk},
            "orig_len_bytes": orig_len_bytes
        });
        watchdog_write_jsonl(dir, "watchdog_encode_steps.jsonl", &rec);
    }
    // tokens_since_reprime removed - using absolute position repriming
    let bar = ProgressBar::new(token_count);
    bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());
    // Backend-specific tuning: SmolLM capped context; RWKV7 can use very long histories
    let effective_context = if backend=="smollm" { cli.context.min(511) } else { usize::MAX };
    // Determine boundaries
    let scan_chunk = agent_chunk; // unify chunking for scan/agent
    let mut next_agent_boundary = agent_chunk; // first agent prime at boundary
    let mut gate_records: Vec<GateRecordV2> = Vec::new();
    let mut total_bits_saved=0.0f64; let mut total_baseline_bits=0.0f64; let mut total_agent_calls: u64=0; let mut total_chunks: u64=0;
    let total_scan_chunks: usize = if scan_enabled { ((token_count as usize) + scan_chunk - 1) / scan_chunk } else { 0 };
    // Separate sessions for XE measurement to avoid state interference (used in scan and agent WIP proof)
    let mut scan_session_smol: Option<SmolLmSession> = None;
    let mut scan_session_rwkv: Option<Rwkv7Session> = None;
    if scan_enabled || agent_enabled {
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

    let mut reprime_hold_until: usize = 0; // skip context re-prime while hint should remain effective
    // Track previous chunk results for learning callback (set env vars before next agent call)
    let mut prev_gate = 0;
    let mut prev_bits_saved = 0.0;
    let mut prev_baseline = 0.0;
    let mut prev_cid = 0;
    let mut prev_bid = 0;
    // Load reuse data if in reuse mode
    let reuse_gates = if reuse_scan_mode { 
        let reuse_dir = cli.reuse_scan_dir.as_ref().unwrap();
        load_gate_decisions_from_csv(reuse_dir) 
    } else { std::collections::HashMap::new() };
    let reuse_cache = if reuse_scan_mode { 
        let reuse_dir = cli.reuse_scan_dir.as_ref().unwrap();
        let cache = load_cached_agent_results(reuse_dir);
        eprintln!("[debug] Loaded {} entries from agent cache in {}", cache.len(), reuse_dir.display());
        for (k, v) in &cache {
            eprintln!("[debug] Cache entry: chunk {} -> {} chars", k, v.agent_text.len());
        }
        cache
    } else { std::collections::HashMap::new() };
    for (i, &sym) in ids.iter().skip(1).enumerate() {
        // * Agentic Conditioning: at the start of each chunk boundary (before encoding current token)
        if (agent_enabled || reuse_scan_mode) && (i + 1) == next_agent_boundary {
            // chunk_index is the chunk we're ABOUT TO START (1-indexed)
            // When next_agent_boundary=512, we're starting chunk 1 (tokens 0-511)
            // When next_agent_boundary=1024, we're starting chunk 2 (tokens 512-1023)
            let chunk_index = next_agent_boundary / agent_chunk;
            let prefix_end_byte = if offsets.is_empty() { i } else if i == 0 { 0 } else { offsets[(i - 1).min(offsets.len() - 1)].1 as usize };
            let prefix_text_for_cache = String::from_utf8_lossy(&data[..prefix_end_byte.min(data.len())]).to_string();
            // Reuse cached agent output if available in reuse mode; otherwise run agent
            // When reusing, we must not run the agent at all
            let (agent_text, agent_duration_ms, agent_calls) = if let Some(cached) = reuse_cache.get(&chunk_index) {
                eprintln!("[reuse] [{}] boundary {}/{} -> reusing cached agent result", backend, chunk_index, ((token_count as usize + agent_chunk - 1) / agent_chunk));
                (cached.agent_text.clone(), 0, cached.agent_calls)
            } else if reuse_scan_mode {
                eprintln!("[debug] Looking for chunk {} in reuse_cache with {} entries", chunk_index, reuse_cache.len());
                eprintln!("[reuse] [{}] boundary {}/{} -> no cached agent result; using empty hint", backend, chunk_index, ((token_count as usize + agent_chunk - 1) / agent_chunk));
                (String::new(), 0, 0)
            } else {
                eprintln!("[agent] [{}] boundary {}/{} tokens[..{}] -> agent...", backend, chunk_index, ((token_count as usize + agent_chunk - 1) / agent_chunk), i);
                let max_steps = if backend=="rwkv7" && cli.scan_max_steps == 7 { 10 } else { cli.scan_max_steps };
                // Set environment variables for the Python agent before calling it
                std::env::set_var("CANDLEZIP_INPUT_FILE", input.to_string_lossy().to_string());
                std::env::set_var("CANDLEZIP_CHUNK_INDEX", chunk_index.to_string());
                // Set learning data from PREVIOUS chunk so the callback records correct results
                std::env::set_var("CANDLEZIP_LAST_GATE", prev_gate.to_string());
                std::env::set_var("CANDLEZIP_LAST_BITS_SAVED", format!("{:.6}", prev_bits_saved));
                std::env::set_var("CANDLEZIP_LAST_BASELINE_BITS", format!("{:.6}", prev_baseline));
                std::env::set_var("CANDLEZIP_LAST_CANDIDATE_ID", prev_cid.to_string());
                std::env::set_var("CANDLEZIP_LAST_BUDGET_ID", prev_bid.to_string());
                run_agent_for_chunk(input, &data, prefix_end_byte, chunk_index, &agent_script_path, &cli.scan_python, &cli.scan_mcp_config, max_steps, cli.scan_verbose, &scan_dir, cli.scan_agent_timeout)?
            };
            if agent_duration_ms > 0 { eprintln!("[agent] [{}] boundary {}/{} agent done in {} ms (calls={}), priming...", backend, chunk_index, ((token_count as usize + agent_chunk - 1) / agent_chunk), agent_duration_ms, agent_calls); }
            // Cache agent result only if it improves compression (gate == 1)
            // This prevents caching failed agent results that could contaminate future runs
            // Compute gate decision and select best candidate/budget BEFORE applying any priming
            if let Some(wtr) = csv_writer_opt.as_mut() {
                let chunk_end = i;
                // Build candidates from agent text
                let build_candidates = |s: &str| -> Vec<String> {
                    let normalized: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
                    let mut out = Vec::new();
                    out.push(normalized.clone());
                    let head = if normalized.len() > 2000 { normalized[..2000].to_string() } else { normalized.clone() };
                    out.push(head);
                    let nums: String = normalized.lines().filter(|l| l.chars().any(|c| c.is_ascii_digit())).collect::<Vec<&str>>().join("\n");
                    out.push(nums);
                    let caps: String = normalized.split_whitespace().filter(|w| {
                        let mut chs = w.chars();
                        match chs.next() { Some(c) if c.is_ascii_uppercase() => true, _ => false }
                    }).collect::<Vec<&str>>().join(" ");
                    out.push(caps);
                    out
                };
                let candidates = build_candidates(&agent_text);
                let max_ctx = if backend=="smollm" { session_smol.as_ref().unwrap().max_context_length().saturating_sub(1) } else { session_rwkv.as_ref().unwrap().max_context_length().saturating_sub(1) };
                let budgets: [usize;4] = [max_ctx/8, max_ctx/4, max_ctx/2, (max_ctx*3)/4];
                let total_len = ids.len();
                let end = (chunk_end + cli.scan_lookahead).min(total_len);
                let target_ids = if chunk_end < end { &ids[chunk_end..end] } else { &[][..] };
                if target_ids.is_empty() { continue; }
                let mut best_bits_saved = f64::NEG_INFINITY;
                let mut best_baseline = 0.0f64; let mut best_conditioned = 0.0f64; let mut best_cid = 0usize; let mut best_bid = 2usize; let mut best_hint_len = 0usize;
                if backend=="smollm" {
                    let history_start = chunk_end.saturating_sub(session_smol.as_ref().unwrap().max_context_length().saturating_sub(1).min(chunk_end));
                    let history = &ids[history_start..chunk_end];
                    let baseline_once = cross_entropy_bits_over_span(scan_session_smol.as_mut().unwrap(), history, target_ids, None)?;
                    for (cid, cand) in candidates.iter().enumerate() {
                        let hint_ids_all = tokenize_hint_smol(tok_hf.as_ref().unwrap(), cand, cli.scan_max_hint_tokens).unwrap_or_else(|_| Vec::new());
                        for (bid, &b) in budgets.iter().enumerate() {
                            let hint_slice = if hint_ids_all.len() > b { &hint_ids_all[..b] } else { &hint_ids_all[..] };
                            let conditioned = if hint_slice.is_empty() { baseline_once } else { cross_entropy_bits_over_span(scan_session_smol.as_mut().unwrap(), history, target_ids, Some(hint_slice))? };
                            let saved = baseline_once - conditioned;
                            if saved > best_bits_saved { best_bits_saved = saved; best_baseline = baseline_once; best_conditioned = conditioned; best_cid = cid; best_bid = bid; best_hint_len = hint_slice.len(); }
                        }
                    }
                } else {
                    let history = &ids[0..chunk_end];
                    let baseline_once = cross_entropy_bits_over_span_rwkv(scan_session_rwkv.as_mut().unwrap(), history, target_ids, None, vocab_size)?;
                    for (cid, cand) in candidates.iter().enumerate() {
                        let hint_ids_all = tokenize_hint_rwkv(tok_rwkv.as_ref().unwrap(), cand, cli.scan_max_hint_tokens).unwrap_or_else(|_| Vec::new());
                        for (bid, &b) in budgets.iter().enumerate() {
                            let hint_slice = if hint_ids_all.len() > b { &hint_ids_all[..b] } else { &hint_ids_all[..] };
                            let conditioned = if hint_slice.is_empty() { baseline_once } else { cross_entropy_bits_over_span_rwkv(scan_session_rwkv.as_mut().unwrap(), history, target_ids, Some(hint_slice), vocab_size)? };
                            let saved = baseline_once - conditioned;
                            if saved > best_bits_saved { best_bits_saved = saved; best_baseline = baseline_once; best_conditioned = conditioned; best_cid = cid; best_bid = bid; best_hint_len = hint_slice.len(); }
                        }
                    }
                }
                let percent_saved = if best_baseline>0.0 { best_bits_saved / best_baseline } else { 0.0 };
                // Gate whenever strictly positive improvement; optional thresholds can raise the bar via flags
                let abs_ok = if cli.scan_gate_threshold_abs_bits > 0.0 { best_bits_saved >= cli.scan_gate_threshold_abs_bits } else { true };
                let pct_ok = if cli.scan_gate_threshold_pct > 0.0 { (percent_saved * 100.0) >= cli.scan_gate_threshold_pct } else { true };
                let gate = if best_bits_saved > 0.0 && abs_ok && pct_ok { 1 } else { 0 };
                // Update scratchpad with prefix snapshot and, if gated, the chosen hint for future chunks
                let prefix_snapshot = if offsets.is_empty() {
                    String::new()
                } else {
                    let end_byte = if i == 0 { 0 } else { offsets[(i - 1).min(offsets.len() - 1)].1 as usize };
                    String::from_utf8_lossy(&data[..end_byte.min(data.len())]).to_string()
                };
                // Write prefix snapshot for the CURRENT chunk (no +1), only during encode
                if !reuse_scan_mode {
                    append_scratchpad(&scan_dir, &format!("prefix_chunk_{}", chunk_index), &prefix_snapshot);
                }
                if gate == 1 {
                    let selected_text = &build_candidates(&agent_text)[best_cid];
                    if !reuse_scan_mode {
                        append_scratchpad(&scan_dir, &format!("accepted_hint_chunk_{}_cand{}_bud{}", chunk_index, best_cid, best_bid), selected_text);
                    }
                    // Cache successful agent result only if it improves compression
                    if let Some(dir) = wdog.as_ref() {
                        cache_agent_result(dir, chunk_index, &agent_text, agent_calls);
                    }
                    // Only append to agent memory if gate == 1 to ensure symmetry, and avoid duplicates by ensuring this chunk not already recorded
                    append_agent_memory(&scan_dir, chunk_index, &agent_text);
                    // Also cache the agent result for potential reuse
                    agent_cache_put(input, chunk_index, &prefix_text_for_cache, &agent_text);
                    // Store results for next chunk's learning callback
                    prev_gate = 1;
                    prev_bits_saved = best_bits_saved;
                    prev_baseline = best_baseline;
                    prev_cid = best_cid;
                    prev_bid = best_bid;
                } else {
                    // Clear any existing cached result for this chunk to prevent contamination
                    if let Some(dir) = wdog.as_ref() {
                        let key = agent_cache_key(input, chunk_index, &prefix_text_for_cache);
                        let path = dir.join("agent_cache").join(format!("{}.txt", key));
                        let _ = fs::remove_file(path);
                    }
                    // Also clear regular cache for consistency
                    let cache_dir = agent_cache_dir();
                    let key = agent_cache_key(input, chunk_index, &prefix_text_for_cache);
                    let path = cache_dir.join(format!("{}.txt", key));
                    let _ = fs::remove_file(path);
                    // Store results for next chunk's learning callback
                    prev_gate = 0;
                    prev_bits_saved = best_bits_saved;
                    prev_baseline = best_baseline;
                    prev_cid = best_cid;
                    prev_bid = best_bid;
                }
                gate_records.push(GateRecordV2{ gate, candidate_id: best_cid as u8, budget_id: best_bid as u8 });
                if gate == 1 {
                    // Apply priming using selected candidate and budget deterministically
                    let selected_text = &build_candidates(&agent_text)[best_cid];
                    let hint_ids_all: Vec<u32> = if backend=="smollm" { tokenize_hint_smol(tok_hf.as_ref().unwrap(), selected_text, cli.scan_max_hint_tokens)? } else { tokenize_hint_rwkv(tok_rwkv.as_ref().unwrap(), selected_text, cli.scan_max_hint_tokens)? };
                    let max_ctx = if backend=="smollm" { session_smol.as_ref().unwrap().max_context_length().saturating_sub(1) } else { session_rwkv.as_ref().unwrap().max_context_length().saturating_sub(1) };
                    let budgets = [max_ctx/8, max_ctx/4, max_ctx/2, (max_ctx*3)/4];
                    let budget_val = budgets[best_bid];
                    let mut prime: Vec<u32> = Vec::new();
                    if backend=="smollm" {
                        let history = &ids[..=i];
                        let hist_take = max_ctx.saturating_sub(budget_val).min(history.len());
                        if hist_take > 0 { let hist_start = history.len() - hist_take; prime.extend_from_slice(&history[hist_start..]); }
                        let hint_take = hint_ids_all.len().min(budget_val);
                        if hint_take > 0 { prime.extend_from_slice(&hint_ids_all[..hint_take]); }
                        logits_t = session_smol.as_mut().unwrap().reprime_with_history_and_get_last_logits_tensor(&prime)?;
                    } else {
                        let mut filtered_hist: Vec<u32> = Vec::with_capacity(i + 1);
                        for &t in ids[..=i].iter() { if (t as usize) < vocab_size { filtered_hist.push(t); } }
                        let hist_take = max_ctx.saturating_sub(budget_val).min(filtered_hist.len());
                        if hist_take > 0 { prime.extend_from_slice(&filtered_hist[filtered_hist.len()-hist_take..]); }
                        let hint_take = hint_ids_all.len().min(budget_val);
                        if hint_take > 0 { prime.extend_from_slice(&hint_ids_all[..hint_take]); }
                        logits_t = session_rwkv.as_mut().unwrap().reprime_with_history_and_get_last_logits_tensor(&prime)?;
                    }
                    // Hold context re-prime for upcoming lookahead tokens so hint remains active
                    reprime_hold_until = i.saturating_add(cli.scan_lookahead);
                    // Also record accepted hint in scratchpad (ensure only once per encode)
                    if !reuse_scan_mode {
                        append_scratchpad(&scan_dir, &format!("accepted_hint_applied_chunk_{}_cand{}_bud{}", chunk_index, best_cid, best_bid), selected_text);
                    }
                    if let Some(dir) = wdog.as_ref() {
                        let rec = json!({
                            "phase": "agent_reprime",
                            "i": i,
                            "chunk_index": chunk_index,
                            "hint_tokens": best_hint_len,
                            "candidate_id": best_cid,
                            "budget_id": best_bid,
                            "agent_text_len": agent_text.len(),
                            "agent_calls": agent_calls,
                        });
                        watchdog_write_jsonl(dir, "watchdog_encode_steps.jsonl", &rec);
                    }
                } else {
                    // Gate is zero: DO NOT prime, and DO NOT cache agent output for this chunk to ensure symmetry.
                    if let Some(dir) = wdog.as_ref() {
                        let key = agent_cache_key(input, chunk_index, &prefix_text_for_cache);
                        let path = dir.join("agent_cache").join(format!("{}.txt", key));
                        let _ = fs::remove_file(path);
                    }
                }
                total_bits_saved += best_bits_saved.max(0.0); total_baseline_bits += best_baseline; total_agent_calls += agent_calls as u64; total_chunks += 1;
                let start_tok = i.saturating_sub(agent_chunk);
                
                // SIMDL v1.1: Compute pricing values
                let gate_bits = if gate == 1 { 5 } else { 0 };
                let price_transcript_bits = if !agent_text.is_empty() {
                    compute_transcript_price_bits(&agent_text).unwrap_or(0)
                } else { 0 };
                let price_pointer_bits = if !agent_text.is_empty() {
                    compute_pointer_price_bits(&doc_index, doc_id, 0, agent_text.len()).unwrap_or(0)
                } else { 0 };
                let tool_id_best = if gate == 1 { format!("cand_{}_bud_{}", best_cid, best_bid) } else { "none".to_string() };
                let tool_snapshot_id = format!("snap_{}", chunk_index);
                let args_hash = blake3_bytes_bin16(agent_text.as_bytes());
                let output_hash = blake3_bytes_bin16(agent_text.as_bytes());
                let chunk_id = format!("{}:{}", chunk_id_prefix, chunk_index);
                
                // Extend CSV with SIMDL v1.1 columns
                wtr.write_record(&[ 
                    input.to_string_lossy().to_string(), chunk_index.to_string(), start_tok.to_string(), i.to_string(), 
                    agent_text.len().to_string(), agent_duration_ms.to_string(), 
                    format!("{:.6}", best_baseline), format!("{:.6}" , best_conditioned), 
                    format!("{:.6}", best_bits_saved), 
                    format!("{:.6}", (if best_baseline>0.0 { best_bits_saved / best_baseline } else { 0.0 })*100.0), 
                    agent_calls.to_string(), gate.to_string(), best_cid.to_string(), best_bid.to_string(),
                    // New SIMDL v1.1 columns
                    gate_bits.to_string(), price_transcript_bits.to_string(), price_pointer_bits.to_string(), tool_id_best,
                    tool_snapshot_id, hex16(&args_hash), hex16(&output_hash), domain.clone(), agent_id.clone(), toolset_id.clone(), run_id.clone(), chunk_id
                ])?; wtr.flush()?;

                // Write learning entry only during encode (not reuse)
                if agent_enabled && !reuse_scan_mode {
                    let _ = write_learning_entry(&scan_dir, input, chunk_index, gate, best_bits_saved, best_baseline, best_cid, best_bid, &agent_text);
                }
                if let Some(dir) = wdog.as_ref() {
                    let hint_token_count: usize = best_hint_len;
                    let rec = json!({
                        "phase":"xe_snapshot",
                        "i": i,
                        "chunk_index": chunk_index,
                        "baseline_bits": best_baseline,
                        "conditioned_bits": best_conditioned,
                        "bits_saved": best_bits_saved,
                        "hint_tokens": hint_token_count,
                        "candidate_id": best_cid,
                        "budget_id": best_bid,
                    });
                    watchdog_write_jsonl(dir, "watchdog_encode_steps.jsonl", &rec);
                }
            } else if reuse_scan_mode {
                // In reuse mode, use cached gate decisions and apply priming accordingly
                if let Some(&(cached_gate, cached_cid, cached_bid)) = reuse_gates.get(&chunk_index) {
                    eprintln!("[reuse] Using cached gate decision for encode chunk {}: gate={}, candidate={}, budget={}", chunk_index, cached_gate, cached_cid, cached_bid);
                    if cached_gate == 1 {
                        // Apply priming using cached candidate and budget to match original encode
                        let build_candidates = |s: &str| -> Vec<String> {
                            let normalized: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
                            let mut out = Vec::new();
                            out.push(normalized.clone());
                            let head = if normalized.len() > 2000 { normalized[..2000].to_string() } else { normalized.clone() };
                            out.push(head);
                            let nums: String = normalized.lines().filter(|l| l.chars().any(|c| c.is_ascii_digit())).collect::<Vec<&str>>().join("\n");
                            out.push(nums);
                            let caps: String = normalized.split_whitespace().filter(|w| {
                                let mut chs = w.chars();
                                match chs.next() { Some(c) if c.is_ascii_uppercase() => true, _ => false }
                            }).collect::<Vec<&str>>().join(" ");
                            out.push(caps);
                            out
                        };
                        let selected_text = &build_candidates(&agent_text)[cached_cid as usize];
                        let hint_ids_all: Vec<u32> = if backend=="smollm" { tokenize_hint_smol(tok_hf.as_ref().unwrap(), selected_text, cli.scan_max_hint_tokens)? } else { tokenize_hint_rwkv(tok_rwkv.as_ref().unwrap(), selected_text, cli.scan_max_hint_tokens)? };
                        let max_ctx = if backend=="smollm" { session_smol.as_ref().unwrap().max_context_length().saturating_sub(1) } else { session_rwkv.as_ref().unwrap().max_context_length().saturating_sub(1) };
                        let budgets = [max_ctx/8, max_ctx/4, max_ctx/2, (max_ctx*3)/4];
                        let budget_val = budgets[cached_bid as usize];
                        let mut prime: Vec<u32> = Vec::new();
                        if backend=="smollm" {
                            let history = &ids[..=i];
                            let hist_take = max_ctx.saturating_sub(budget_val).min(history.len());
                            if hist_take > 0 { let hist_start = history.len() - hist_take; prime.extend_from_slice(&history[hist_start..]); }
                            let hint_take = hint_ids_all.len().min(budget_val);
                            if hint_take > 0 { prime.extend_from_slice(&hint_ids_all[..hint_take]); }
                            logits_t = session_smol.as_mut().unwrap().reprime_with_history_and_get_last_logits_tensor(&prime)?;
                        } else {
                            let mut filtered_hist: Vec<u32> = Vec::with_capacity(i + 1);
                            for &t in ids[..=i].iter() { if (t as usize) < vocab_size { filtered_hist.push(t); } }
                            let hist_take = max_ctx.saturating_sub(budget_val).min(filtered_hist.len());
                            if hist_take > 0 { prime.extend_from_slice(&filtered_hist[filtered_hist.len()-hist_take..]); }
                            let hint_take = hint_ids_all.len().min(budget_val);
                            if hint_take > 0 { prime.extend_from_slice(&hint_ids_all[..hint_take]); }
                            logits_t = session_rwkv.as_mut().unwrap().reprime_with_history_and_get_last_logits_tensor(&prime)?;
                        }
                        // Hold context re-prime for upcoming lookahead tokens so hint remains active
                        reprime_hold_until = i.saturating_add(cli.scan_lookahead);
                    }
                }
            }
            next_agent_boundary += agent_chunk;
        }
        // Do not append a human-readable summary into the scratchpad during encode.
        // Keeping the scratchpad minimal (prefix snapshots + accepted hints) avoids
        // noisy repeated blocks being shown in agent prompts/logs.
        // Reprime logic - deterministic based on absolute position only
        if backend=="smollm" {
            // SmolLM: Reprime when session.index_pos >= effective_context AND i % reprime_interval == 0 (absolute position)
            let session = session_smol.as_ref().unwrap();
            if i < reprime_hold_until {
                // skip reprime to preserve agent hint influence
            } else if session.index_pos() >= effective_context && (i % cli.reprime_interval) == 0 && i > 0 {
                let end = 1 + i;
                let start = end.saturating_sub(effective_context);
                let history = &ids[start..end];
                logits_t = session_smol.as_mut().unwrap().reprime_with_history_and_get_last_logits_tensor(history)?;
                if let Some(dir) = wdog.as_ref() {
                    let rec = json!({"phase":"context_reprime","i":i,"window":effective_context});
                    watchdog_write_jsonl(dir, "watchdog_encode_steps.jsonl", &rec);
                }
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

        if let Some(dir) = wdog.as_ref() {
            // Record stable per-step digest without leaking logits.
            let p_digest = {
                if backend=="smollm" {
                    let vec = logits_t.to_vec1::<f32>()?; // Won't be written directly
                    hex16(&blake3_f32_bin16(&vec))
                } else {
                    let vec = logits_t.to_vec1::<f32>()?;
                    hex16(&blake3_f32_bin16(&vec))
                }
            };
            let rec = json!({
                "phase": "encode_step",
                "i": i,
                "sym": sym,
                "c_lo": c_lo,
                "c_hi": c_hi,
                "pdf_digest": p_digest,
                "bytes_out": ace.bytes_written(),
            });
            watchdog_write_jsonl(dir, "watchdog_encode_steps.jsonl", &rec);
        }

        if backend=="smollm" {
            logits_t = session_smol.as_mut().unwrap().step_logits_tensor(sym)?;
        } else {
            if (sym as usize) < vocab_size {
                logits_t = session_rwkv.as_mut().unwrap().step_logits_tensor(sym)?;
            }
        }
        // tokens_since_reprime removed - using absolute position repriming

        // Old scan block moved to agent boundary above
        if i % 1024 == 0 { bar.set_position(i as u64); let bytes_so_far = ace.bytes_written(); let bpb = (8.0 * bytes_so_far as f64) / (orig_len_bytes as f64); bar.set_message(format!("bytes={}  bpb={:.3}", bytes_so_far, bpb)); }
    }
    bar.finish_and_clear();
    // Finish encoder and retrieve payload bytes
    let payload_bytes = ace.finish()?;
    if let Some(dir) = wdog.as_ref() {
        let rec = json!({"phase":"encode_done","bytes_out": payload_bytes.len()});
        watchdog_write_jsonl(dir, "watchdog_encode_steps.jsonl", &rec);
    }

    // Build structured gating records per boundary (candidate + budget + gate)
    let gates_present = agent_enabled;
    // Write final file: header -> optional agent gates -> payload
    {
        let mut out_file = BufWriter::new(File::create(output)?);
        let reserved_flags = flags_pack(agent_enabled, agent_mock, gates_present, agent_chunk);
        let header_v2 = HeaderBinV2 { bos_token_id: bos_id, token_count, orig_len_bytes, model_hash16, tokenizer_hash16, orig_hash16: orig_blake3_16, reserved_flags, context_window: cli.context as u32, vocab_size: vocab_size as u32, model_file_repr_len: weight_repr.len() as u32, reprime_interval: cli.reprime_interval as u32 };
        write_header_v2(&mut out_file, &header_v2, weight_repr.as_bytes())?;
        if gates_present { write_agent_gates_v2(&mut out_file, &gate_records)?; }
        use std::io::Write as _;
        out_file.write_all(&payload_bytes)?;
        out_file.flush()?;
    }

    let enc_bytes = fs::metadata(output)?.len() as u64; let elapsed = t0.elapsed(); let bpb = (8.0 * enc_bytes as f64) / (orig_len_bytes as f64); let char_count = String::from_utf8_lossy(&data).chars().count() as u64; let bpc = if char_count > 0 { (8.0 * enc_bytes as f64) / (char_count as f64) } else { f64::NAN };
    println!("Encoded: {} bytes -> {} bytes | bits/byte={:.3} | bits/char={:.3} | context={} | time={:.2?}", orig_len_bytes, enc_bytes, bpb, bpc, cli.context, elapsed);
    if let Some(dir) = wdog.as_ref() {
        let rec = json!({
            "phase":"summary",
            "orig_len_bytes": orig_len_bytes,
            "enc_len_bytes": enc_bytes,
            "bits_per_byte": bpb,
            "bits_per_char": bpc,
            "elapsed_sec": elapsed.as_secs_f64()
        });
        watchdog_write_jsonl(dir, "watchdog_encode_steps.jsonl", &rec);
    }
    if scan_enabled || agent_enabled { let mut meta=scan_meta; if let Some(obj)=meta.as_object_mut() { obj.insert("orig_len_bytes".to_string(), json!(orig_len_bytes)); obj.insert("encoded_len_bytes".to_string(), json!(enc_bytes)); obj.insert("bits_per_byte".to_string(), json!(bpb)); obj.insert("bits_per_char".to_string(), json!(bpc)); obj.insert("elapsed_sec".to_string(), json!(elapsed.as_secs_f64())); obj.insert("scan_total_chunks".to_string(), json!(total_chunks)); obj.insert("scan_total_agent_calls".to_string(), json!(total_agent_calls)); obj.insert("scan_total_bits_saved".to_string(), json!(total_bits_saved)); let percent_overall = if total_baseline_bits>0.0 { total_bits_saved/total_baseline_bits } else { 0.0 }; obj.insert("scan_percent_saved_overall".to_string(), json!(percent_overall*100.0)); } let meta_path=scan_dir.join("meta.json"); fs::write(meta_path, serde_json::to_vec_pretty(&meta)?)?; }

    // Cleanup agent memory for this run (ensure no cross-run contamination)
    if agent_enabled {
        let mem_dir = scan_dir.join("agent_mem");
        let _ = fs::remove_dir_all(mem_dir);
    }
    
    // Save SIMDL v1.1 document index 
    if let Err(e) = doc_index.save_to_path(&Path::new("index_meta.json")) {
        eprintln!("Warning: Failed to save document index: {}", e);
    }
    
    Ok(())
}

fn decode_file(cli: &Cli, input: &Path, output: &Path) -> anyhow::Result<()> {
    let reuse_scan_mode = cli.reuse_scan_dir.is_some();
    let device = detect_device(cli.cpu);
    let mut rdr = BufReader::new(File::open(input)?);
    let (header, model_file_repr) = read_header_v2(&mut rdr)?;
    let backend = cli.backend.to_lowercase();
    let vocab_size = header.vocab_size as usize;
    let agent_in_header = flags_unpack_agent_used(header.reserved_flags);
    let agent_mock_in_header = flags_unpack_agent_mock(header.reserved_flags);
    let agent_chunk = flags_unpack_agent_chunk(header.reserved_flags).max(1);
    let agent_script_path = cli.scan_agent_script.clone();
    let is_mock_script = agent_script_path.file_name().and_then(|s| s.to_str()).unwrap_or("") == "deterministic_mock_agent_cli.py";
    if agent_in_header {
        if !cli.agent { anyhow::bail!("file requires --agent for deterministic decoding"); }
        if agent_mock_in_header && !is_mock_script { anyhow::bail!("file encoded with deterministic mock agent; pass --scan-agent-script agent/deterministic_mock_agent_cli.py"); }
        if !agent_mock_in_header && is_mock_script { anyhow::bail!("file encoded with real agent; deterministic mock agent cannot be used for decoding"); }
    }
    // Prepare optional log dir when agent is required
    let mut run_dir: Option<PathBuf> = None;
    if agent_in_header || cli.scan || reuse_scan_mode {
        if let Some(reuse_dir) = &cli.reuse_scan_dir {
            // Reuse existing scan directory
            run_dir = Some(reuse_dir.clone());
            if !reuse_dir.exists() { anyhow::bail!("Reuse scan directory does not exist: {}", reuse_dir.display()); }
            eprintln!("[reuse] Using existing scan directory for decode: {}", reuse_dir.display());
        } else {
            let ts = Utc::now().format("%Y%m%d_%H%M%S");
            let run_dir_name = format!("{}_decode_{}_{}", input.file_stem().and_then(|s| s.to_str()).unwrap_or("input"), backend, ts);
            let mut dir = cli.scan_output_dir.join(run_dir_name);
            if let Ok(override_dir) = std::env::var("CANDLEZIP_WATCHDOG_DIR") {
                dir = PathBuf::from(override_dir);
            }
            fs::create_dir_all(&dir).ok();
            run_dir = Some(dir);
        }
    }

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
        // * Enforce agent usage consistency if present in header
        if agent_in_header && !cli.agent {
            anyhow::bail!("file requires --agent for deterministic decoding");
        }
        let tokenizer = HfTokenizer::from_file(tok_path.to_str().unwrap()).map_err(|e| anyhow::anyhow!("failed loading tokenizer: {e}"))?;
        let mut session = SmolLmSession::load(&weight_paths, &config_path, device, vocab_size)?;
        // Optional gating section directly after header (supports v1 bits or v2 structured)
        let mut gates_bytes: Vec<u8> = Vec::new();
        let mut gate_records: Vec<GateRecordV2> = Vec::new();
        if flags_unpack_agent_gates(header.reserved_flags) {
            let mut magic = [0u8;4];
            rdr.read_exact(&mut magic)?;
            if magic == AGT2_MAGIC {
                let count = read_var_u64(&mut rdr)? as usize;
                let mut buf = vec![0u8; count];
                rdr.read_exact(&mut buf)?;
                for b in buf { gate_records.push(GateRecordV2{ gate: (b>>0)&1, candidate_id: (b>>1)&0b11, budget_id: (b>>3)&0b11 }); }
            } else if magic == AGTB_MAGIC {
                let nbits = read_var_u64(&mut rdr)?;
                let nbytes = ((nbits + 7) / 8) as usize;
                let mut bytes = vec![0u8; nbytes];
                rdr.read_exact(&mut bytes)?;
                gates_bytes = bytes;
            } else { anyhow::bail!("invalid gating magic"); }
        }
        let mut payload = Vec::new(); rdr.read_to_end(&mut payload)?; let mut acd = ArithmeticDecoder::new(&payload[..])?; let mut logits_t = session.step_logits_tensor(header.bos_token_id)?;
        let wdog = watchdog_dir();
        // Load reuse data if in reuse mode
        let reuse_gates = if reuse_scan_mode { load_gate_decisions_from_csv(run_dir.as_ref().unwrap()) } else { std::collections::HashMap::new() };
        let agent_cache = if reuse_scan_mode {
            load_cached_agent_results(run_dir.as_ref().unwrap())
        } else if cli.reuse {
            if let Some(dir) = wdog.as_ref() {
                load_cached_agent_results(dir)
            } else {
                std::collections::HashMap::new()
            }
        } else {
            std::collections::HashMap::new()
        };
        if let Some(dir) = wdog.as_ref() {
            let rec = json!({"phase":"decode_init","backend":"smollm","bos":header.bos_token_id,"reuse":cli.reuse});
            watchdog_write_jsonl(dir, "watchdog_decode_steps.jsonl", &rec);
        }
        let mut out_tokens: Vec<u32> = Vec::with_capacity(header.token_count as usize + 1); out_tokens.push(header.bos_token_id);
        let bar = ProgressBar::new(header.token_count); bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());
        let effective_context = (header.context_window as usize).saturating_sub(1).min(511);
        // tokens_since_reprime removed - using absolute position repriming
        let mut next_agent_boundary = agent_chunk;
        // Load gates: prefer structured v2; else bits; else fallback to proof.csv in watchdog dir
        let mut gate_decisions: Vec<u8> = Vec::new();
        if !gate_records.is_empty() {
            // use structured
        } else if !gates_bytes.is_empty() {
            let total_bits = (gates_bytes.len() * 8) as usize;
            for i in 0..total_bits { gate_decisions.push(get_gate_bit(&gates_bytes, i)); }
        } else if let Some(dir) = watchdog_dir().as_ref() {
            let proof_path = Path::new(dir).join("proof.csv");
            if let Ok(txt) = fs::read_to_string(&proof_path) {
                for (li, line) in txt.lines().enumerate() {
                    if li == 0 { continue; }
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 12 { if let Ok(g) = parts[11].trim().parse::<u8>() { gate_decisions.push(g); } }
                }
            }
        }
        let mut first_mismatch_written = false;
        let mut reprime_hold_until: u64 = 0; // skip context re-prime while hint should remain effective
        for i in 0..header.token_count {
            // Context reprime if needed - deterministic based on absolute position only  
            if i < reprime_hold_until {
                // skip reprime to preserve agent hint influence
            } else if session.index_pos() >= effective_context && (i % (header.reprime_interval as u64)) == 0 && i > 0 {
                let end = out_tokens.len(); 
                let start = end.saturating_sub(effective_context); 
                let history = &out_tokens[start..end]; 
                logits_t = session.reprime_with_history_and_get_last_logits_tensor(history)?; 
                if let Some(dir) = wdog.as_ref() {
                    let rec = json!({"phase":"context_reprime","i":i,"window":effective_context});
                    watchdog_write_jsonl(dir, "watchdog_decode_steps.jsonl", &rec);
                }
            }
            // Agent priming at boundary, before decoding current token
            if agent_in_header && (i + 1) == next_agent_boundary as u64 {
                let chunk_index = next_agent_boundary / agent_chunk;
                let (gate, cand_id, bud_id) = if let Some(&(cached_gate, cached_cid, cached_bid)) = reuse_gates.get(&chunk_index) {
                    // Use cached gate decision from CSV
                    eprintln!("[reuse] Using cached gate decision for decode chunk {}: gate={}, candidate={}, budget={}", chunk_index, cached_gate, cached_cid, cached_bid);
                    (cached_gate, cached_cid as u8, cached_bid as u8)
                } else if !gate_records.is_empty() {
                    let rec = gate_records.get(chunk_index.saturating_sub(1)).copied().unwrap_or_default();
                    (rec.gate, rec.candidate_id, rec.budget_id)
                } else {
                    let g = gate_decisions.get(chunk_index.saturating_sub(1)).copied().unwrap_or_else(|| get_gate_bit(&gates_bytes, chunk_index.saturating_sub(1)));
                    (g, 0, 2)
                };
                if gate == 1 {
                    let agent_text = if cli.reuse || reuse_scan_mode {
                        if let Some(cached) = agent_cache.get(&chunk_index) {
                            eprintln!("[agent] [smollm] boundary {}/{} -> reusing cached agent result", chunk_index, ((header.token_count as usize + agent_chunk - 1) / agent_chunk));
                            cached.agent_text.clone()
                        } else if cli.reuse {
                            anyhow::bail!("--reuse specified but no cached agent result for chunk {}", chunk_index);
                        } else {
                            eprintln!("[reuse] [smollm] boundary {}/{} -> no cached agent result; using empty hint", chunk_index, ((header.token_count as usize + agent_chunk - 1) / agent_chunk));
                            String::new()
                        }
                    } else {
                        let prefix_text = tokenizer.decode(&out_tokens[1..], true).map_err(|e| anyhow::anyhow!("tokenizer.decode failed: {e}"))?;
                        let (agent_text, _dur_ms, _calls) = run_agent_for_prefix_text(
                            &prefix_text,
                            chunk_index,
                            &agent_script_path,
                            &cli.scan_python,
                            &cli.scan_mcp_config,
                            if cli.scan_max_steps == 7 { 10 } else { cli.scan_max_steps },
                            cli.scan_verbose,
                            run_dir.as_deref().unwrap_or_else(|| Path::new("scan_output")),
                            cli.scan_agent_timeout,
                        )?;
                        agent_text
                    };
                    // Record agent text to memory only when gate == 1 to ensure symmetry
                    if let Some(dir) = run_dir.as_ref() {
                        append_agent_memory(dir, chunk_index, &agent_text);
                    }
                    // Deterministically construct candidates and apply selected id and budget
                    let build_candidates = |s: &str| -> Vec<String> {
                        let normalized: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
                        let mut out = Vec::new();
                        out.push(normalized.clone());
                        let head = if normalized.len() > 2000 { normalized[..2000].to_string() } else { normalized.clone() };
                        out.push(head);
                        let nums: String = normalized.lines().filter(|l| l.chars().any(|c| c.is_ascii_digit())).collect::<Vec<&str>>().join("\n");
                        out.push(nums);
                        let caps: String = normalized.split_whitespace().filter(|w| { let mut chs = w.chars(); match chs.next() { Some(c) if c.is_ascii_uppercase() => true, _ => false } }).collect::<Vec<&str>>().join(" ");
                        out.push(caps);
                        out
                    };
                    let candidates = build_candidates(&agent_text);
                    let selected = candidates.get(cand_id as usize).cloned().unwrap_or_else(|| agent_text.clone());
                    let hint_ids = tokenize_hint_smol(&tokenizer, &selected, cli.scan_max_hint_tokens)?;
                    let max_ctx = session.max_context_length().saturating_sub(1);
                    let budgets = [max_ctx/8, max_ctx/4, max_ctx/2, (max_ctx*3)/4];
                    let budget_val = budgets.get(bud_id as usize).copied().unwrap_or(max_ctx/4);
                    let mut prime: Vec<u32> = Vec::new();
                    // History first, then hint (must match encode)
                    let history = &out_tokens[..];
                    let hist_take = max_ctx.saturating_sub(budget_val).min(history.len());
                    if hist_take > 0 { let hist_start = history.len().saturating_sub(hist_take); prime.extend_from_slice(&history[hist_start..]); }
                    let hint_take = hint_ids.len().min(budget_val);
                    if hint_take > 0 { prime.extend_from_slice(&hint_ids[..hint_take]); }
                    logits_t = session.reprime_with_history_and_get_last_logits_tensor(&prime)?;
                    // Hold context re-prime for upcoming lookahead tokens so hint remains active
                    reprime_hold_until = i.saturating_add(cli.scan_lookahead as u64);
                } else {
                    eprintln!("[agent] [smollm] Skipping agent priming for chunk {} (gate=0)", chunk_index);
                }
                // tokens_since_reprime removed - using absolute position repriming
                next_agent_boundary += agent_chunk;
            }
            // Compute pdf from current logits
            let logits_vec = logits_t.to_vec1::<f32>()?; let pdf = softmax_pdf_floor(&logits_vec, vocab_size, ac_p_min());
            // Decode symbol and step
            let sym = acd.decode_symbol(&pdf)? as u32; out_tokens.push(sym); logits_t = session.step_logits_tensor(sym)?; if i % 1024 == 0 { bar.set_position(i); }
            if let Some(dir) = wdog.as_ref() {
                // Compare against encode trace if available to pinpoint first differing symbol
                let enc_trace = watchdog_load_encode_trace(dir);
                if (i as usize) < enc_trace.len() {
                    if let Some(enc) = &enc_trace[i as usize] {
                        if enc.sym != sym && !first_mismatch_written {
                            first_mismatch_written = true;
                            let rec = json!({
                                "phase":"mismatch",
                                "i": i,
                                "encode_sym": enc.sym,
                                "decode_sym": sym,
                                "note": "first differing symbol"
                            });
                            watchdog_try_write_json(dir, "watchdog_mismatch.json", &rec);
                        }
                    }
                }
            }
            if let Some(dir) = wdog.as_ref() {
                let p_digest = {
                    let vec = logits_t.to_vec1::<f32>()?;
                    hex16(&blake3_f32_bin16(&vec))
                };
                let rec = json!({"phase":"decode_step","i":i,"sym":sym,"pdf_digest":p_digest});
                watchdog_write_jsonl(dir, "watchdog_decode_steps.jsonl", &rec);
            }
        }
        bar.finish_and_clear(); let detok = tokenizer.decode(&out_tokens[1..], true).map_err(|e| anyhow::anyhow!("tokenizer.decode failed: {e}"))?; fs::write(output, detok.as_bytes())?;
        if let Some(dir) = wdog.as_ref() {
            let rec = json!({"phase":"decode_done","tokens": out_tokens.len()});
            watchdog_write_jsonl(dir, "watchdog_decode_steps.jsonl", &rec);
        }
        // Cleanup agent memory dir if created
        if agent_in_header {
            if let Some(dir) = run_dir {
                let _ = fs::remove_dir_all(dir.join("agent_mem"));
            }
        }
        Ok(())
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
        if agent_in_header && !cli.agent {
            anyhow::bail!("file requires --agent for deterministic decoding");
        }
        let tokenizer = candlerwkv7::models::rwkv7::Tokenizer::new(&tok_path)?;
        let mut session = Rwkv7Session::load(&model_path, &config_path, device)?;
        // Optional gating section directly after header
        let mut gates_bytes: Vec<u8> = Vec::new();
        let mut gates_v2: Vec<GateRecordV2> = Vec::new();
        if flags_unpack_agent_gates(header.reserved_flags) {
            let mut magic = [0u8;4];
            rdr.read_exact(&mut magic)?;
            if magic == AGT2_MAGIC {
                let count = read_var_u64(&mut rdr)? as usize;
                let mut buf = vec![0u8; count];
                rdr.read_exact(&mut buf)?;
                for b in buf { gates_v2.push(GateRecordV2{ gate: (b>>0)&1, candidate_id: (b>>1)&0b11, budget_id: (b>>3)&0b11 }); }
            } else if magic == AGTB_MAGIC {
                let nbits = read_var_u64(&mut rdr)?;
                let nbytes = ((nbits + 7) / 8) as usize;
                let mut bytes = vec![0u8; nbytes];
                rdr.read_exact(&mut bytes)?;
                gates_bytes = bytes;
            } else { anyhow::bail!("invalid gating magic"); }
        }
        let mut payload = Vec::new();
        rdr.read_to_end(&mut payload)?;
        let mut acd = ArithmeticDecoder::new(&payload[..])?;
        let mut logits_t = session.step_logits_tensor(header.bos_token_id)?;
        let mut out_syms: Vec<u32> = Vec::with_capacity(header.token_count as usize + 1);
        out_syms.push(header.bos_token_id);
        let bar = ProgressBar::new(header.token_count);
        bar.set_style(ProgressStyle::with_template("{wide_bar} {pos}/{len}").unwrap());
        let mut next_agent_boundary = agent_chunk;
        let wdog = watchdog_dir();
        // Load reuse data if in reuse mode
        let reuse_gates = if reuse_scan_mode { load_gate_decisions_from_csv(run_dir.as_ref().unwrap()) } else { std::collections::HashMap::new() };
        let agent_cache = if reuse_scan_mode {
            load_cached_agent_results(run_dir.as_ref().unwrap())
        } else if cli.reuse {
            if let Some(dir) = wdog.as_ref() {
                load_cached_agent_results(dir)
            } else {
                std::collections::HashMap::new()
            }
        } else {
            std::collections::HashMap::new()
        };
        if let Some(dir) = wdog.as_ref() {
            let rec = json!({"phase":"decode_init","backend":"rwkv7","bos":header.bos_token_id,"reuse":cli.reuse});
            watchdog_write_jsonl(dir, "watchdog_decode_steps.jsonl", &rec);
        }
        let mut first_mismatch_written = false;
        let mut reprime_hold_until: u64 = 0; // skip context re-prime while hint should remain effective
        // RWKV7 decoding with proper literal handling and agent priming
        for i in 0..header.token_count {
            // Agent priming at boundary
            if agent_in_header && (i + 1) == next_agent_boundary as u64 {
                let chunk_index = next_agent_boundary / agent_chunk;
                let (gate, cand_id, bud_id) = if let Some(&(cached_gate, cached_cid, cached_bid)) = reuse_gates.get(&chunk_index) {
                    // Use cached gate decision from CSV
                    eprintln!("[reuse] Using cached gate decision for decode chunk {}: gate={}, candidate={}, budget={}", chunk_index, cached_gate, cached_cid, cached_bid);
                    (cached_gate, cached_cid as u8, cached_bid as u8)
                } else if !gates_v2.is_empty() {
                    let rec = gates_v2.get(chunk_index.saturating_sub(1)).copied().unwrap_or_default();
                    (rec.gate, rec.candidate_id, rec.budget_id)
                } else {
                    (get_gate_bit(&gates_bytes, chunk_index.saturating_sub(1)), 0, 2)
                };
                if gate == 1 {
                    let agent_text = if cli.reuse || reuse_scan_mode {
                        if let Some(cached) = agent_cache.get(&chunk_index) {
                            eprintln!("[agent] [rwkv7] boundary {}/{} -> reusing cached agent result", chunk_index, ((header.token_count as usize + agent_chunk - 1) / agent_chunk));
                            cached.agent_text.clone()
                        } else if cli.reuse {
                            anyhow::bail!("--reuse specified but no cached agent result for chunk {}", chunk_index);
                        } else {
                            eprintln!("[reuse] [rwkv7] boundary {}/{} -> no cached agent result; using empty hint", chunk_index, ((header.token_count as usize + agent_chunk - 1) / agent_chunk));
                            String::new()
                        }
                    } else {
                        let detok_bytes_prefix = rwkv_detok_with_literals(&tokenizer, &out_syms[1..], session.vocab_size());
                        let prefix_text = String::from_utf8_lossy(&detok_bytes_prefix).to_string();
                        let (agent_text, _dur_ms, _calls) = run_agent_for_prefix_text(
                            &prefix_text, chunk_index, &agent_script_path, &cli.scan_python, &cli.scan_mcp_config,
                            if cli.scan_max_steps == 7 { 10 } else { cli.scan_max_steps }, cli.scan_verbose,
                            run_dir.as_deref().unwrap_or_else(|| Path::new("scan_output")), cli.scan_agent_timeout,
                        )?;
                        agent_text
                    };
                    // Record agent text to memory only when gate == 1 to ensure symmetry
                    if let Some(dir) = run_dir.as_ref() {
                        append_agent_memory(dir, chunk_index, &agent_text);
                    }
                    // Deterministically pick candidate and budget (from metadata)
                    let build_candidates = |s: &str| -> Vec<String> {
                        let normalized: String = s.chars().map(|c| if c.is_control() { ' ' } else { c }).collect();
                        let mut out = Vec::new();
                        out.push(normalized.clone());
                        let head = if normalized.len() > 2000 { normalized[..2000].to_string() } else { normalized.clone() };
                        out.push(head);
                        let nums: String = normalized.lines().filter(|l| l.chars().any(|c| c.is_ascii_digit())).collect::<Vec<&str>>().join("\n");
                        out.push(nums);
                        let caps: String = normalized.split_whitespace().filter(|w| { let mut chs = w.chars(); match chs.next() { Some(c) if c.is_ascii_uppercase() => true, _ => false } }).collect::<Vec<&str>>().join(" ");
                        out.push(caps);
                        out
                    };
                    let candidates = build_candidates(&agent_text);
                    let selected = candidates.get(cand_id as usize).cloned().unwrap_or_else(|| agent_text.clone());
                    let hint_ids = tokenize_hint_rwkv(&tokenizer, &selected, cli.scan_max_hint_tokens)?;
                    // Build prime: history + hint (match encode ordering)
                    let max_ctx = session.max_context_length().saturating_sub(1);
                    let budgets = [max_ctx/8, max_ctx/4, max_ctx/2, (max_ctx*3)/4];
                    let budget_val = budgets.get(bud_id as usize).copied().unwrap_or(max_ctx/4);
                    let mut prime: Vec<u32> = Vec::new();
                    // Filter out literals from history, then take history first up to (max_ctx - budget_val)
                    let mut filtered_hist: Vec<u32> = Vec::with_capacity(out_syms.len());
                    for &t in out_syms.iter() { if (t as usize) < session.vocab_size() { filtered_hist.push(t); } }
                    let hist_take = max_ctx.saturating_sub(budget_val).min(filtered_hist.len());
                    if hist_take > 0 { prime.extend_from_slice(&filtered_hist[filtered_hist.len().saturating_sub(hist_take)..]); }
                    let hint_take = hint_ids.len().min(budget_val);
                    if hint_take > 0 { prime.extend_from_slice(&hint_ids[..hint_take]); }
                    logits_t = session.reprime_with_history_and_get_last_logits_tensor(&prime)?;
                    // Hold context re-prime for upcoming lookahead tokens so hint remains active
                    reprime_hold_until = i.saturating_add(cli.scan_lookahead as u64);
                } else {
                    eprintln!("[agent] [rwkv7] Skipping agent priming for chunk {} (gate=0)", chunk_index);
                }
                next_agent_boundary += agent_chunk;
            }

            let logits_vec = logits_t.to_vec1::<f32>()?;
            let pdf = combined_pdf_with_literals(&logits_vec, session.vocab_size());
            let sym = acd.decode_symbol(&pdf)? as u32;
            out_syms.push(sym);

            // Only step the model if it's a valid token (not literal)
            if (sym as usize) < session.vocab_size() {
                logits_t = session.step_logits_tensor(sym)?;
            }

            if let Some(dir) = wdog.as_ref() {
                let vec = logits_t.to_vec1::<f32>()?;
                let rec = json!({"phase":"decode_step","i":i,"sym":sym,"pdf_digest":hex16(&blake3_f32_bin16(&vec))});
                watchdog_write_jsonl(dir, "watchdog_decode_steps.jsonl", &rec);
                // Compare against encode trace (if present) to mark first differing symbol deterministically
                let enc_trace = watchdog_load_encode_trace(dir);
                if (i as usize) < enc_trace.len() {
                    if let Some(enc) = &enc_trace[i as usize] {
                        if enc.sym != sym && !first_mismatch_written {
                            first_mismatch_written = true;
                            let rec = json!({
                                "phase":"mismatch",
                                "i": i,
                                "encode_sym": enc.sym,
                                "decode_sym": sym,
                                "note": "first differing symbol"
                            });
                            watchdog_try_write_json(dir, "watchdog_mismatch.json", &rec);
                        }
                    }
                }
            }
            if i % 256 == 0 { bar.set_position(i); }
        }
        bar.finish_and_clear();

        // Convert combined symbol stream back to bytes
        let detok_bytes = rwkv_detok_with_literals(&tokenizer, &out_syms[1..], session.vocab_size());
        fs::write(output, &detok_bytes)?;
        if let Some(dir) = wdog.as_ref() {
            let rec = json!({"phase":"decode_done","symbols": out_syms.len()});
            watchdog_write_jsonl(dir, "watchdog_decode_steps.jsonl", &rec);
        }
        // Cleanup agent memory dir if created
        if agent_in_header {
            if let Some(dir) = run_dir {
                let _ = fs::remove_dir_all(dir.join("agent_mem"));
            }
        }
        Ok(())
    } else {
        anyhow::bail!("unknown backend {}", backend);
    }
}



