// RWKV7 Chat Example
// Interactive chat interface with temperature, top-p, and other sampling options

use anyhow::Result;
use candlerwkv7::models::rwkv7::{Config, Model, State, Tokenizer};
use candle::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use clap::Parser;
use std::io::{self, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(author, version, about = "Interactive chat with RWKV7")]
struct Args {
    /// Path to the .pth model file (rwkv7-g1-0.1b-20250307-ctx4096.pth)
    #[arg(long, default_value = "modeldir/rwkv7-g1-0.1b-20250307-ctx4096.pth")]
    model_pth: PathBuf,

    /// Path to the vocabulary file
    #[arg(long, default_value = "modeldir/rwkv_vocab_v20230424.json")]
    vocab: PathBuf,

    /// Use CPU instead of GPU
    #[arg(long)]
    cpu: bool,

    /// Initial prompt/system message
    #[arg(long, default_value = "User: Hello! What can you help me with today?\nAssistant:")]
    prompt: String,

    /// The temperature used to generate samples (0.0 = greedy, higher = more random)
    #[arg(long, default_value = "0.1")]
    temperature: f64,

    /// Nucleus sampling probability cutoff
    #[arg(long, default_value = "0.9")]
    top_p: f64,

    /// The seed to use when generating random samples
    #[arg(long, default_value = "42")]
    seed: u64,

    /// Maximum length of the response (in tokens)
    #[arg(long, default_value = "200")]
    max_length: usize,

    /// Interactive mode (continue conversation)
    #[arg(long)]
    interactive: bool,

    /// Show generation statistics
    #[arg(long)]
    verbose: bool,
}

struct ChatSession {
    model: Model,
    tokenizer: Tokenizer,
    state: State,
    logits_processor: LogitsProcessor,
    device: Device,
    config: Config,
    conversation_history: String,
    conversation_tokens: Vec<u32>,
    last_token_idx: usize,
}

impl ChatSession {
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        config: Config,
        device: Device,
        seed: u64,
        temperature: f64,
        top_p: f64,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, Some(temperature), Some(top_p));
        let state = State::new(1, &config, None, &device).unwrap();
        
        Self {
            model,
            tokenizer,
            state,
            logits_processor,
            device,
            config,
            conversation_history: String::new(),
            conversation_tokens: Vec::new(),
            last_token_idx: 0,
        }
    }

    fn generate_response(&mut self, input_text: &str, max_length: usize, verbose: bool) -> Result<String> {
        // Append raw text for readability
        self.conversation_history.push_str(input_text);

        // Tokenize only the new input and append tokens to conversation_tokens
        let new_tokens = self.tokenizer.encode(input_text)?;
        if verbose {
            println!("New tokens: {}", new_tokens.len());
        }
        let start_time = std::time::Instant::now();

        // Append new tokens and process only them to update state incrementally
        let prev_len = self.conversation_tokens.len();
        self.conversation_tokens.extend(new_tokens.iter().cloned());
        let mut final_logits: Option<Tensor> = None;
        for &token in &self.conversation_tokens[prev_len..] {
            let input_tensor = Tensor::new(&[[token]], &self.device)?;
            let logits = self.model.forward(&input_tensor, &mut self.state)?;
            final_logits = Some(logits);
        }
        self.last_token_idx = self.conversation_tokens.len();
        
        // Generate new tokens
        let mut response_tokens = Vec::new();
        let mut current_logits = final_logits;
        
        for step in 0..max_length {
            let logits = match current_logits.as_ref() {
                Some(logits) => logits,
                None => anyhow::bail!("No logits available"),
            };
            
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            
            // Check for NaN/Inf
            let min_val = logits.min(0)?.to_scalar::<f32>()?;
            let max_val = logits.max(0)?.to_scalar::<f32>()?;
            if min_val.is_nan() || min_val.is_infinite() || max_val.is_nan() || max_val.is_infinite() {
                println!("Warning: NaN/Inf detected at step {}", step);
                break;
            }
            
            // Sample next token
            let next_token = self.logits_processor.sample(&logits)?;
            response_tokens.push(next_token);
            // append generated token to conversation_tokens so state remains consistent
            self.conversation_tokens.push(next_token);
            self.last_token_idx += 1;
            
            // Check for stop conditions
            if is_stop_token(next_token) {
                break;
            }
            
            // Forward pass for next iteration
            let input_tensor = Tensor::new(&[[next_token]], &self.device)?;
            current_logits = Some(self.model.forward(&input_tensor, &mut self.state)?);
        }
        
        let generation_time = start_time.elapsed();
        
        // Decode response
        let response = self.tokenizer.decode(&response_tokens)?;
        
        // Add response to conversation history (text form)
        self.conversation_history.push_str(&response);
        
        if verbose {
            println!("Generated {} tokens in {:?} ({:.2} tokens/sec)", 
                response_tokens.len(), 
                generation_time,
                response_tokens.len() as f64 / generation_time.as_secs_f64());
        }
        
        Ok(response)
    }
    
    fn reset_conversation(&mut self) {
        self.conversation_history.clear();
        self.state = State::new(1, &self.config, None, &self.device).unwrap();
    }
}

fn is_stop_token(token: u32) -> bool {
    // Common stop tokens (adjust based on your tokenizer)
    matches!(token, 0 | 1 | 2) // Usually EOS, BOS, UNK
}

fn load_model(model_pth: &PathBuf, vocab: &PathBuf, device: &Device) -> Result<(Model, Tokenizer, Config)> {
    // Ensure conversion
    let prepared_dir = PathBuf::from("modeldir/prepared");
    std::fs::create_dir_all(&prepared_dir)?;
    
    let model_safetensors = prepared_dir.join("model.safetensors");
    let config_json = prepared_dir.join("config.json");
    
    if !model_safetensors.exists() || !config_json.exists() {
        println!("Converting model from .pth to safetensors format...");
        let status = std::process::Command::new("python")
            .arg("convert_pth_direct.py")
            .arg("--src").arg(model_pth)
            .arg("--dest").arg(&model_safetensors)
            .arg("--config").arg(&config_json)
            .status()?;
        
        if !status.success() {
            anyhow::bail!("Failed to convert model");
        }
    }
    
    // Load tokenizer and model
    let tokenizer = Tokenizer::new(vocab)?;
    let config: Config = serde_json::from_str(&std::fs::read_to_string(&config_json)?)?;
    
    println!("Loading model...");
    let tensors = candle::safetensors::load(&model_safetensors, device)?;
    let vb = VarBuilder::from_tensors(tensors, DType::F32, device);
    let model = Model::new(&config, vb)?;
    
    Ok((model, tokenizer, config))
}

fn main() -> Result<()> {
    let args = Args::parse();
    
    let device = if args.cpu { Device::Cpu } else { Device::new_cuda(0)? };
    println!("Using device: {:?}", device);
    println!("Temperature: {}, Top-p: {}, Seed: {}", args.temperature, args.top_p, args.seed);
    
    // Load model
    let (model, tokenizer, config) = load_model(&args.model_pth, &args.vocab, &device)?;
    
    // Create chat session
    let mut chat = ChatSession::new(
        model, 
        tokenizer, 
        config, 
        device, 
        args.seed, 
        args.temperature, 
        args.top_p
    );
    
    if args.interactive {
        println!("RWKV7 Interactive Chat");
        println!("Type 'quit' to exit, 'reset' to clear conversation, 'help' for commands");
        println!("{}", "=".repeat(60));
        
        loop {
            print!("\nYou: ");
            io::stdout().flush()?;
            
            let mut input = String::new();
            io::stdin().read_line(&mut input)?;
            let input = input.trim();
            
            match input {
                "quit" | "exit" => break,
                "reset" => {
                    chat.reset_conversation();
                    println!("Conversation reset.");
                    continue;
                }
                "help" => {
                    println!("Commands:");
                    println!("  quit/exit - Exit the chat");
                    println!("  reset     - Clear conversation history");
                    println!("  help      - Show this help");
                    continue;
                }
                "" => continue,
                _ => {}
            }
            
            let prompt = format!("User: {}\nAssistant:", input);
            
            print!("Assistant:");
            io::stdout().flush()?;
            
            match chat.generate_response(&prompt, args.max_length, args.verbose) {
                Ok(response) => {
                    println!("{}", response);
                }
                Err(e) => {
                    println!("Error: {}", e);
                }
            }
        }
    } else {
        // Single response mode
        println!("Prompt: {}", args.prompt);
        print!("Response:");
        io::stdout().flush()?;
        
        let response = chat.generate_response(&args.prompt, args.max_length, args.verbose)?;
        println!("{}", response);
    }
    
    Ok(())
}
