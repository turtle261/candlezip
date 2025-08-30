use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ArgGroup, ValueHint};

use negent::{
    DEFAULT_MODEL_PATH, DEFAULT_CONFIG_PATH, DEFAULT_TOKENIZER_PATH,
    compute_compressed_bits_bytes, compute_compressed_bits, ncd, message_distance_bits,
    compute_compressed_bits_verbose, compute_compressed_bits_bytes_verbose,
};

#[derive(Parser, Debug)]
#[command(name = "negent", about = "Kolmogorov complexity approximation (RWKV7-based) — bit-length, conditional length, NCD")]
struct Cli {
    /// Force running on CPU (default tries CUDA 0 then CPU)
    #[arg(long, global = true)]
    cpu: bool,

    /// Verbose progress: show candlezip-like progress bars and stats
    #[arg(long, global = true)]
    verbose: bool,

    /// Path to model weights (.safetensors file)
    #[arg(long, default_value = DEFAULT_MODEL_PATH, value_hint = ValueHint::FilePath, global = true)]
    model: PathBuf,

    /// Path to config.json file
    #[arg(long, default_value = DEFAULT_CONFIG_PATH, value_hint = ValueHint::FilePath, global = true)]
    config: PathBuf,

    /// Path to tokenizer.json (local)
    #[arg(long, default_value = DEFAULT_TOKENIZER_PATH, value_hint = ValueHint::FilePath, global = true)]
    tokenizer: PathBuf,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Compute C(x): compressed bit-length
    Bits(OneInput),

    /// Compute C(y|x): conditional compressed bit-length
    BitsCond(TwoInputs),

    /// Compute NCD(x,y)
    Ncd(TwoInputsTextOrFile),

    /// Compute message distance D(x,y) = C(y|x) + C(x|y) in bits
    Distance(TwoInputsTextOrFile),
}

#[derive(Parser, Debug)]
#[command(group(ArgGroup::new("input").required(true).args(["text", "file"])))]
struct OneInput {
    /// Text input X
    #[arg(long, group = "input")]
    text: Option<String>,

    /// File input X
    #[arg(long, group = "input", value_hint = ValueHint::FilePath)]
    file: Option<PathBuf>,

    /// Optional proof mode (roundtrip verify). Slower, but mathematically verifies coder correctness.
    #[arg(long)]
    proof: bool,
}

#[derive(Parser, Debug)]
#[command(group(ArgGroup::new("x").required(true).args(["text_x", "file_x"])))]
#[command(group(ArgGroup::new("y").required(true).args(["text_y", "file_y"])))]
struct TwoInputs {
    /// Text input X
    #[arg(long, group = "x")]
    text_x: Option<String>,
    /// File input X
    #[arg(long, group = "x", value_hint = ValueHint::FilePath)]
    file_x: Option<PathBuf>,

    /// Text input Y
    #[arg(long, group = "y")]
    text_y: Option<String>,
    /// File input Y
    #[arg(long, group = "y", value_hint = ValueHint::FilePath)]
    file_y: Option<PathBuf>,

    /// Optional proof mode (roundtrip verify). Slower, but mathematically verifies coder correctness.
    #[arg(long)]
    proof: bool,
}

#[derive(Parser, Debug)]
#[command(group(ArgGroup::new("x").required(true).args(["text_x", "file_x"])))]
#[command(group(ArgGroup::new("y").required(true).args(["text_y", "file_y"])))]
struct TwoInputsTextOrFile {
    /// Text input X
    #[arg(long, group = "x")]
    text_x: Option<String>,
    /// File input X
    #[arg(long, group = "x", value_hint = ValueHint::FilePath)]
    file_x: Option<PathBuf>,

    /// Text input Y
    #[arg(long, group = "y")]
    text_y: Option<String>,
    /// File input Y
    #[arg(long, group = "y", value_hint = ValueHint::FilePath)]
    file_y: Option<PathBuf>,
}

fn load_input_text_or_bytes(text: &Option<String>, file: &Option<PathBuf>) -> Result<(Option<String>, Vec<u8>)> {
    if let Some(t) = text { return Ok((Some(t.clone()), t.as_bytes().to_vec())); }
    let p = file.as_ref().expect("group enforces one of text/file");
    let bytes = fs::read(p).with_context(|| format!("failed to read {}", p.display()))?;
    Ok((None, bytes))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Bits(args) => {
            let (maybe_text, bytes) = load_input_text_or_bytes(&args.text, &args.file)?;
            if let Some(t) = maybe_text {
                let bits = if cli.verbose {
                    compute_compressed_bits_verbose(&t, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof, "C(x)")?
                } else {
                    compute_compressed_bits(&t, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof)?
                };
                println!("{}", bits);
            } else {
                let bits = if cli.verbose {
                    compute_compressed_bits_bytes_verbose(&bytes, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof, "C(x)")?
                } else {
                    compute_compressed_bits_bytes(&bytes, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof)?
                };
                println!("{}", bits);
            }
        }
        Commands::BitsCond(args) => {
            let (maybe_text_x, bytes_x) = load_input_text_or_bytes(&args.text_x, &args.file_x)?;
            let (maybe_text_y, bytes_y) = load_input_text_or_bytes(&args.text_y, &args.file_y)?;
            // We prefer using strings when both are strings (to avoid extra encode/decode), otherwise bytes
            let bits = match (maybe_text_x, maybe_text_y) {
                (Some(x), Some(y)) => {
                    if cli.verbose {
                        compute_compressed_bits_verbose(&y, Some(&x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof, "C(y|x)")?
                    } else {
                        negent::compute_compressed_bits(&y, Some(&x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof)?
                    }
                }
                _ => {
                    if cli.verbose {
                        compute_compressed_bits_bytes_verbose(&bytes_y, Some(&bytes_x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof, "C(y|x)")?
                    } else {
                        negent::compute_compressed_bits_bytes(&bytes_y, Some(&bytes_x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, args.proof)?
                    }
                }
            };
            println!("{}", bits);
        }
        Commands::Ncd(args) => {
            let (maybe_text_x, bytes_x) = load_input_text_or_bytes(&args.text_x, &args.file_x)?;
            let (maybe_text_y, bytes_y) = load_input_text_or_bytes(&args.text_y, &args.file_y)?;
            let val = match (maybe_text_x, maybe_text_y) {
                (Some(x), Some(y)) => {
                    if cli.verbose {
                        let _cx = compute_compressed_bits_verbose(&x, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(x)")? as f64;
                        let _cy = compute_compressed_bits_verbose(&y, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(y)")? as f64;
                        let c_xy = compute_compressed_bits_verbose(&y, Some(&x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(y|x)")? as f64;
                        let c_yx = compute_compressed_bits_verbose(&x, Some(&y), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(x|y)")? as f64;
                        // Recompute c_x, c_y quickly without verbose to avoid duplicate bars
                        let c_x = negent::compute_compressed_bits(&x, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        let c_y = negent::compute_compressed_bits(&y, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        (c_xy + c_yx) / (c_x + c_y)
                    } else {
                        ncd(&x, &y, &cli.model, &cli.config, &cli.tokenizer, cli.cpu)?
                    }
                }
                _ => {
                    if cli.verbose {
                        let _cx = compute_compressed_bits_bytes_verbose(&bytes_x, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(x)")? as f64;
                        let _cy = compute_compressed_bits_bytes_verbose(&bytes_y, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(y)")? as f64;
                        let c_xy = compute_compressed_bits_bytes_verbose(&bytes_y, Some(&bytes_x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(y|x)")? as f64;
                        let c_yx = compute_compressed_bits_bytes_verbose(&bytes_x, Some(&bytes_y), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(x|y)")? as f64;
                        // Quick non-verbose totals to compute denominator
                        let c_x = negent::compute_compressed_bits_bytes(&bytes_x, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        let c_y = negent::compute_compressed_bits_bytes(&bytes_y, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        (c_xy + c_yx) / (c_x + c_y)
                    } else {
                        let c_x = negent::compute_compressed_bits_bytes(&bytes_x, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        let c_y = negent::compute_compressed_bits_bytes(&bytes_y, None, &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        let c_xy = negent::compute_compressed_bits_bytes(&bytes_y, Some(&bytes_x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        let c_yx = negent::compute_compressed_bits_bytes(&bytes_x, Some(&bytes_y), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)? as f64;
                        (c_xy + c_yx) / (c_x + c_y)
                    }
                }
            };
            println!("{}", val);
        }
        Commands::Distance(args) => {
            let (maybe_text_x, bytes_x) = load_input_text_or_bytes(&args.text_x, &args.file_x)?;
            let (maybe_text_y, bytes_y) = load_input_text_or_bytes(&args.text_y, &args.file_y)?;
            let bits: u64 = match (maybe_text_x, maybe_text_y) {
                (Some(x), Some(y)) => {
                    if cli.verbose {
                        let c_xy = compute_compressed_bits_verbose(&y, Some(&x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(y|x)")?;
                        let c_yx = compute_compressed_bits_verbose(&x, Some(&y), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(x|y)")?;
                        c_xy + c_yx
                    } else {
                        message_distance_bits(&x, &y, &cli.model, &cli.config, &cli.tokenizer, cli.cpu)?
                    }
                }
                _ => {
                    if cli.verbose {
                        let c_xy = compute_compressed_bits_bytes_verbose(&bytes_y, Some(&bytes_x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(y|x)")?;
                        let c_yx = compute_compressed_bits_bytes_verbose(&bytes_x, Some(&bytes_y), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false, "C(x|y)")?;
                        c_xy + c_yx
                    } else {
                        let c_xy = negent::compute_compressed_bits_bytes(&bytes_y, Some(&bytes_x), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)?;
                        let c_yx = negent::compute_compressed_bits_bytes(&bytes_x, Some(&bytes_y), &cli.model, &cli.config, &cli.tokenizer, cli.cpu, false)?;
                        c_xy + c_yx
                    }
                }
            };
            println!("{}", bits);
        }
    }
    Ok(())
}
