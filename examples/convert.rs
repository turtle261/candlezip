// Model Conversion Utility
// Converts RWKV7 .pth files to safetensors format using the Python script

use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Convert RWKV7 .pth files to safetensors format")]
struct Args {
    /// Path to the source .pth file
    #[arg(long)]
    src: PathBuf,

    /// Path for the output .safetensors file
    #[arg(long)]
    dest: PathBuf,

    /// Path for the output config.json file
    #[arg(long)]
    config: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    println!("Converting RWKV7 model from {} to {}", args.src.display(), args.dest.display());
    
    // Ensure the Python conversion script exists
    let python_script = PathBuf::from("convert_pth_direct.py");
    if !python_script.exists() {
        anyhow::bail!("Python conversion script not found: {}", python_script.display());
    }
    
    // Prepare command arguments
    let mut cmd = std::process::Command::new("python");
    cmd.arg(&python_script)
       .arg("--src").arg(&args.src)
       .arg("--dest").arg(&args.dest);
    
    if let Some(config_path) = &args.config {
        cmd.arg("--config").arg(config_path);
    }
    
    // Run the conversion
    println!("Running: python {} --src {} --dest {}", 
        python_script.display(), args.src.display(), args.dest.display());
    
    let status = cmd.status()?;
    
    if status.success() {
        println!("✅ Conversion completed successfully!");
        if args.dest.exists() {
            let metadata = std::fs::metadata(&args.dest)?;
            println!("   Output file size: {} bytes", metadata.len());
        }
        if let Some(config_path) = &args.config {
            if config_path.exists() {
                println!("   Config file created: {}", config_path.display());
            }
        }
    } else {
        anyhow::bail!("❌ Conversion failed with exit code: {:?}", status.code());
    }
    
    Ok(())
}
