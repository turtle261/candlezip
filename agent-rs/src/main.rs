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

use anyhow::{anyhow, Result};
use clap::{Arg, Command};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    env,
    path::Path,
    time::{Duration, Instant},
};
use tracing::{debug, info, warn};

// Rig + RMCP (for deterministic tool-using agent with MCP tools)
use rig::{agent::AgentBuilder, providers::gemini};
use rig::client::{ProviderClient, CompletionClient};
use rig::completion::Prompt;
use rmcp::{
    model::{ClientCapabilities, ClientInfo, Implementation, ProtocolVersion},
    service::{ServiceExt as _, RunningService},
    transport::child_process::TokioChildProcess,
};
use tokio::process::Command as TokioCommand;
use tokio::time::sleep;

/// Agent response with timing information
#[derive(Debug, Serialize, Deserialize)]
struct AgentResponse {
    final_text: String,
    duration_ms: u64,
}

/// Deterministic agent configuration
#[derive(Debug, Clone)]
struct DeterministicAgentConfig {
    model: String,
    temperature: f64,
    max_tokens: Option<u64>,
    max_steps: usize,
    timeout_ms: u64,
    seed: Option<u64>,
}

impl Default for DeterministicAgentConfig {
    fn default() -> Self {
        Self {
            model: "gemini-2.5-flash".to_string(),
            temperature: 0.0,
            max_tokens: Some(2048),
            max_steps: 7,
            timeout_ms: 300_000, // 5 minutes
            seed: Some(42),
        }
    }
}

/// MCP server configuration
#[derive(Debug, Deserialize)]
struct McpServerConfig {
    command: String,
    args: Vec<String>,
    env: HashMap<String, String>,
}

/// MCP configuration file structure
#[derive(Debug, Deserialize)]
struct McpConfig {
    #[serde(rename = "mcpServers")]
    mcp_servers: HashMap<String, McpServerConfig>,
}

/// Load MCP configuration from file
fn load_mcp_config<P: AsRef<Path>>(path: P) -> Result<Option<McpConfig>> {
    let path = path.as_ref();
    if !path.exists() {
        warn!("MCP config file not found: {}", path.display());
        return Ok(None);
    }

    let content = std::fs::read_to_string(path)?;
    let config: McpConfig = serde_json::from_str(&content)?;
    Ok(Some(config))
}

/// Gemini API request structures
#[derive(Debug, Serialize)]
struct GeminiRequest {
    contents: Vec<GeminiContent>,
    #[serde(rename = "generationConfig")]
    generation_config: GeminiGenerationConfig,
    #[serde(rename = "safetySettings")]
    safety_settings: Vec<GeminiSafetySetting>,
}

#[derive(Debug, Serialize)]
struct GeminiContent {
    parts: Vec<GeminiPart>,
    role: String,
}

#[derive(Debug, Serialize)]
struct GeminiPart {
    text: String,
}

#[derive(Debug, Serialize)]
struct GeminiGenerationConfig {
    temperature: f64,
    #[serde(rename = "maxOutputTokens")]
    max_output_tokens: Option<u64>,
    #[serde(rename = "candidateCount")]
    candidate_count: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    seed: Option<u64>,
}

#[derive(Debug, Serialize)]
struct GeminiSafetySetting {
    category: String,
    threshold: String,
}

/// Gemini API response structures
#[derive(Debug, Deserialize)]
struct GeminiResponse {
    candidates: Option<Vec<GeminiCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
    error: Option<GeminiError>,
}

#[derive(Debug, Deserialize)]
struct GeminiError {
    code: Option<i32>,
    message: String,
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiResponseContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponseContent {
    parts: Option<Vec<GeminiResponsePart>>,
}

#[derive(Debug, Deserialize)]
struct GeminiResponsePart {
    text: String,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u64>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u64>,
    #[serde(rename = "totalTokenCount")]
    total_token_count: Option<u64>,
}

/// Create a deterministic Gemini client
fn create_gemini_client() -> Result<Client> {
    let client = Client::builder()
        .timeout(Duration::from_millis(300_000))
        .build()?;
    Ok(client)
}

/// Call Gemini API with deterministic settings
async fn call_gemini_api(
    client: &Client,
    task: &str,
    config: &DeterministicAgentConfig,
) -> Result<String> {
    // Ensure API key is available
    let api_key = env::var("GEMINI_API_KEY")
        .or_else(|_| env::var("GOOGLE_API_KEY"))
        .map_err(|_| anyhow!("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required"))?;

    // Prepare the system prompt for compression context
    let system_prompt = "You are a capable AI assistant that helps with text prediction for compression. \
        You approach tasks methodically and provide clear, precisely accurate responses. \
        You pay close attention to detail and provide exact information that will appear \
        later in the text supplied. You understand that your information will need to \
        reduce uncertainty on prediction of text! You make it clear where the beginning \
        and end of your response is--include any noise that is likely associated!";

    let full_prompt = format!("{}\n\n{}", system_prompt, task);

    // Create request
    let request = GeminiRequest {
        contents: vec![GeminiContent {
            parts: vec![GeminiPart {
                text: full_prompt,
            }],
            role: "user".to_string(),
        }],
        generation_config: GeminiGenerationConfig {
            temperature: config.temperature,
            max_output_tokens: config.max_tokens,
            candidate_count: 1,
            seed: config.seed,
        },
        safety_settings: vec![
            GeminiSafetySetting {
                category: "HARM_CATEGORY_HARASSMENT".to_string(),
                threshold: "BLOCK_NONE".to_string(),
            },
            GeminiSafetySetting {
                category: "HARM_CATEGORY_HATE_SPEECH".to_string(),
                threshold: "BLOCK_NONE".to_string(),
            },
            GeminiSafetySetting {
                category: "HARM_CATEGORY_SEXUALLY_EXPLICIT".to_string(),
                threshold: "BLOCK_NONE".to_string(),
            },
            GeminiSafetySetting {
                category: "HARM_CATEGORY_DANGEROUS_CONTENT".to_string(),
                threshold: "BLOCK_NONE".to_string(),
            },
        ],
    };

    // Build URL - use the correct endpoint
    let url = format!(
        "https://generativelanguage.googleapis.com/v1/models/{}:generateContent?key={}",
        config.model, api_key
    );

    // Make request
    let response = client
        .post(&url)
        .json(&request)
        .send()
        .await?;

    // Check if request was successful
    if !response.status().is_success() {
        let error_text = response.text().await?;
        return Err(anyhow!("Gemini API error: {}", error_text));
    }

    // Get the response text for debugging
    let response_text = response.text().await?;
    debug!("Gemini API response: {}", response_text);

    // Parse response - try to be more flexible with error handling
    let gemini_response: Result<GeminiResponse, _> = serde_json::from_str(&response_text);

    match gemini_response {
        Ok(response) => {
            // Check if there's an API error in the response
            if let Some(error) = response.error {
                return Err(anyhow!("Gemini API error: {} (status: {:?}, code: {:?})",
                    error.message, error.status, error.code));
            }

            // Extract text from response
            if let Some(candidates) = response.candidates {
                if let Some(candidate) = candidates.first() {
                    if let Some(parts) = &candidate.content.parts {
                        if let Some(part) = parts.first() {
                            Ok(part.text.clone())
                        } else {
                            Err(anyhow!("No text parts in Gemini response"))
                        }
                    } else {
                        Err(anyhow!("No parts in Gemini response content"))
                    }
                } else {
                    Err(anyhow!("No candidates in Gemini response"))
                }
            } else {
                Err(anyhow!("No candidates field in Gemini response"))
            }
        }
        Err(parse_error) => {
            // If parsing fails, try to extract error message from response
            if let Ok(error_response) = serde_json::from_str::<serde_json::Value>(&response_text) {
                if let Some(error) = error_response.get("error") {
                    if let Some(message) = error.get("message") {
                        return Err(anyhow!("Gemini API error: {}", message));
                    }
                    return Err(anyhow!("Gemini API error: {}", error));
                }
            }
            Err(anyhow!("Failed to parse Gemini response: {} - Response: {}", parse_error, response_text))
        }
    }
}

/// Spawn MCP servers from config and return connected services and their sink peers
async fn connect_mcp_servers(
    cfg: &McpConfig,
) -> Result<(Vec<RunningService<rmcp::RoleClient, ClientInfo>>, Vec<rmcp::service::ServerSink>)> {
    let mut running: Vec<RunningService<rmcp::RoleClient, ClientInfo>> = Vec::new();
    let mut sinks: Vec<rmcp::service::ServerSink> = Vec::new();

    for (_name, spec) in &cfg.mcp_servers {
        // Build child process transport (stdio)
        let mut cmd = TokioCommand::new(&spec.command);
        for a in &spec.args {
            cmd.arg(a);
        }
        // Merge env
        for (k, v) in &spec.env {
            cmd.env(k, v);
        }

        // Spawn stdio transport
        let transport = TokioChildProcess::new(cmd)?;

        // Identify as deterministic client
        let client_info = ClientInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ClientCapabilities::default(),
            client_info: Implementation::from_build_env(),
        };

        let service = client_info
            .serve(transport)
            .await
            .inspect_err(|e| warn!("MCP client error: {:?}", e))?;

        sinks.push(service.peer().clone());
        running.push(service);
    }

    Ok((running, sinks))
}

/// Build a deterministic Rig agent with optional MCP tools and run the task
async fn run_agent_task(
    task: &str,
    mcp_config_path: Option<&str>,
    config: &DeterministicAgentConfig,
) -> Result<AgentResponse> {
    let start_time = Instant::now();

    // Ensure API key is available early
    if env::var("GEMINI_API_KEY").is_err() && env::var("GOOGLE_API_KEY").is_err() {
        return Err(anyhow!("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required"));
    }

    // Build Gemini model client
    let gemini_client = gemini::Client::from_env();

    // Deterministic additional params: block no content, top-p 1.0, single candidate
    let deterministic_params = serde_json::json!({
        "generationConfig": {
            "candidateCount": 1,
            "topP": 1.0,
            // Pass a deterministic seed when available; some providers honor this
            "seed": config.seed,
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    });

    // System preamble for compression context (matches prior behavior)
    let system_preamble = "You are a capable AI assistant that helps with text prediction for compression. \
        You approach tasks methodically and provide clear, precisely accurate responses. \
        You pay close attention to detail and provide exact information that will appear \
        later in the text supplied. You understand that your information will need to \
        reduce uncertainty on prediction of text! You make it clear where the beginning \
        and end of your response is--include any noise that is likely associated!";

    // Create agent builder
    let mut builder = AgentBuilder::new(
        gemini_client.completion_model(&config.model),
    )
    .preamble(system_preamble)
    .temperature(config.temperature)
    .max_tokens(config.max_tokens.unwrap_or(2048))
    .additional_params(deterministic_params);

    // Optional MCP integration
    let mut _services: Vec<RunningService<rmcp::RoleClient, ClientInfo>> = Vec::new();
    if let Some(path) = mcp_config_path {
        if let Some(cfg) = load_mcp_config(path)? {
            let (services, sinks) = connect_mcp_servers(&cfg).await?;
            // keep services alive for the duration of the run
            _services = services;
            for sink in sinks.into_iter() {
                // enumerate tools and attach
                let tools = sink.list_tools(Default::default()).await?.tools;
                for t in tools {
                    builder = builder.rmcp_tool(t, sink.clone());
                }
            }
        }
    }

    // Build agent and execute with multi_turn for tool use
    let agent = builder.build();

    // Retry wrapper to mitigate transient HTTP decode issues from providers
    let mut attempt: u32 = 0;
    let output = loop {
        attempt += 1;
        let run = async {
            let output = agent
                .prompt(task)
                .multi_turn(config.max_steps)
                .await
                .map_err(|e| anyhow!("Agent run failed: {}", e))?;
            Ok::<String, anyhow::Error>(output)
        };

        match tokio::time::timeout(Duration::from_millis(config.timeout_ms), run).await {
            Ok(Ok(s)) => break s,
            Ok(Err(e)) => {
                let es = format!("{}", e);
                // Known flaky path: reqwest HttpError decoding body
                if attempt < 5 && (es.contains("HttpError") || es.contains("error decoding response body") || es.contains("timeout") || es.contains("connection reset") || es.contains("EOF")) {
                    warn!("Attempt {} failed: {}. Retrying...", attempt, es);
                    sleep(Duration::from_millis(750 * attempt as u64)).await;
                    continue;
                } else {
                    return Err(e);
                }
            }
            Err(_) => {
                if attempt < 5 {
                    warn!("Attempt {} timed out after {}ms. Retrying...", attempt, config.timeout_ms);
                    continue;
                } else {
                    return Err(anyhow!("Agent task timed out after {}ms", config.timeout_ms));
                }
            }
        }
    };

    let duration = start_time.elapsed();

    Ok(AgentResponse { final_text: output, duration_ms: duration.as_millis() as u64 })
}

/// Set up deterministic environment
fn setup_deterministic_environment() {
    // Set deterministic environment variables
    unsafe {
        std::env::set_var("TOKENIZERS_PARALLELISM", "false");
        std::env::set_var("PYTHONHASHSEED", "0");
        std::env::set_var("TZ", "UTC");
        std::env::set_var("LANG", "C");
        std::env::set_var("LC_ALL", "C");
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    // Set up deterministic environment
    setup_deterministic_environment();

    // Load environment variables
    dotenv::dotenv().ok();

    // Parse command line arguments
    let matches = Command::new("agent-rs")
        .about("Deterministic Rig-based agent for lossless compression")
        .version("0.1.0")
        .arg(
            Arg::new("task")
                .long("task")
                .value_name("TASK")
                .help("Task description for the agent to complete")
                .required(true),
        )
        .arg(
            Arg::new("model")
                .long("model")
                .value_name("MODEL")
                .help("Model name to use (Gemini)")
                .default_value("gemini-2.5-flash"),
        )
        .arg(
            Arg::new("mcp-config")
                .long("mcp-config")
                .value_name("PATH")
                .help("Path to MCP configuration JSON file")
                .default_value("mcp_config.json"),
        )
        .arg(
            Arg::new("max-steps")
                .long("max-steps")
                .value_name("STEPS")
                .help("Maximum reasoning/tool steps for the agent")
                .value_parser(clap::value_parser!(usize))
                .default_value("7"),
        )
        .arg(
            Arg::new("timeout")
                .long("timeout")
                .value_name("MS")
                .help("Timeout in milliseconds")
                .value_parser(clap::value_parser!(u64))
                .default_value("300000"),
        )
        .arg(
            Arg::new("temperature")
                .long("temperature")
                .value_name("TEMP")
                .help("Temperature for the model (should be 0.0 for determinism)")
                .value_parser(clap::value_parser!(f64))
                .default_value("0.0"),
        )
        .get_matches();

    let task = matches.get_one::<String>("task").unwrap();
    let model = matches.get_one::<String>("model").unwrap();
    let mcp_config_path = matches.get_one::<String>("mcp-config");
    let max_steps = *matches.get_one::<usize>("max-steps").unwrap();
    let timeout_ms = *matches.get_one::<u64>("timeout").unwrap();
    let temperature = *matches.get_one::<f64>("temperature").unwrap();

    // Warn if temperature is not 0.0
    if temperature != 0.0 {
        warn!("Temperature is set to {}, but 0.0 is recommended for determinism", temperature);
    }

    let config = DeterministicAgentConfig {
        model: model.clone(),
        temperature,
        max_steps,
        timeout_ms,
        ..Default::default()
    };

    info!("Running deterministic agent with config: {:?}", config);
    info!("Task: {}", task);

    // Run the agent
    let response = run_agent_task(
        task,
        mcp_config_path.map(|s| s.as_str()),
        &config,
    ).await?;

    // Output in compatible format
    let json_output = serde_json::json!({
        "final_text": response.final_text,
        "duration_ms": response.duration_ms
    });

    println!("AGENT_RESULT_JSON:{}", json_output);
    println!("{}", response.final_text);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Test that the same input produces the same output (determinism)
    #[tokio::test]
    async fn test_deterministic_output() -> Result<()> {
        // Skip test if no API key is available
        if env::var("GEMINI_API_KEY").is_err() && env::var("GOOGLE_API_KEY").is_err() {
            println!("Skipping test: GEMINI_API_KEY or GOOGLE_API_KEY not set");
            return Ok(());
        }

        setup_deterministic_environment();

        let config = DeterministicAgentConfig::default();
        let task = "Analyze this text prefix and predict what content is likely to appear next: 'The quick brown fox jumps over the lazy'";

        // Run the same task multiple times
        let mut responses = Vec::new();
        for i in 0..3 {
            println!("Running test iteration {}", i + 1);
            let response = run_agent_task(task, None, &config).await?;
            responses.push(response.final_text);
            
            // Add a small delay to ensure different timestamps
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        // Check that all responses are identical
        let unique_responses: HashSet<_> = responses.iter().collect();
        
        println!("Unique responses: {}", unique_responses.len());
        for (i, response) in responses.iter().enumerate() {
            println!("Response {}: {}", i + 1, response);
        }

        // For true determinism, we should have exactly 1 unique response
        assert_eq!(
            unique_responses.len(),
            1,
            "Agent should produce identical outputs for identical inputs. Got {} unique responses",
            unique_responses.len()
        );

        Ok(())
    }

    /// Test that different inputs produce different outputs
    #[tokio::test]
    async fn test_different_inputs_different_outputs() -> Result<()> {
        // Skip test if no API key is available
        if env::var("GEMINI_API_KEY").is_err() && env::var("GOOGLE_API_KEY").is_err() {
            println!("Skipping test: GEMINI_API_KEY or GOOGLE_API_KEY not set");
            return Ok(());
        }

        setup_deterministic_environment();

        let config = DeterministicAgentConfig::default();
        
        let task1 = "Predict what comes next: 'The weather today is'";
        let task2 = "Predict what comes next: 'The recipe calls for'";

        let response1 = run_agent_task(task1, None, &config).await?;
        let response2 = run_agent_task(task2, None, &config).await?;

        // Different inputs should produce different outputs
        assert_ne!(
            response1.final_text,
            response2.final_text,
            "Different inputs should produce different outputs"
        );

        Ok(())
    }

    /// Test agent configuration validation
    #[test]
    fn test_config_validation() {
        let config = DeterministicAgentConfig::default();
        
        assert_eq!(config.temperature, 0.0, "Default temperature should be 0.0 for determinism");
        assert_eq!(config.seed, Some(42), "Default seed should be set for determinism");
        assert!(config.max_tokens.is_some(), "Max tokens should be set");
        assert!(config.timeout_ms > 0, "Timeout should be positive");
    }

    /// Test MCP config loading
    #[test]
    fn test_mcp_config_loading() -> Result<()> {
        // Test with non-existent file
        let config = load_mcp_config("non_existent.json")?;
        assert!(config.is_none());

        // Test with the actual MCP config if it exists
        if Path::new("../agent/mcp_config.json").exists() {
            let config = load_mcp_config("../agent/mcp_config.json")?;
            assert!(config.is_some());
            
            let config = config.unwrap();
            assert!(!config.mcp_servers.is_empty());
        }

        Ok(())
    }
}
