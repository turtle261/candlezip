# agent-rs

A deterministic Rust-based agent implementation using Google's Gemini API, designed for lossless compression applications where reproducible outputs are critical.

## Overview

This agent is a replacement for the Python CrewAI-based agent used in CandleZip compression. It provides deterministic text prediction capabilities by directly calling the Gemini API with carefully controlled parameters:

- **Temperature: 0.0** for maximum determinism
- **Seed: 42** for reproducible outputs
- **Consistent prompting** with compression-focused instructions
- **Timeout handling** for robust production use

## Features

- **Deterministic Output**: Same input always produces the same output
- **Direct Gemini API**: No heavy dependencies on agent frameworks
- **Command-line Interface**: Compatible with existing compression pipeline
- **Comprehensive Testing**: Validates determinism with multiple test runs
- **MCP Configuration Support**: Ready for future tool integration

## Installation & Build

```bash
cd agent-rs
cargo build --release
```

## Usage

### Basic Usage

```bash
# Run with a text prediction task
./target/release/agent-rs --task "Predict what comes next: 'The quick brown fox'"

# With custom parameters
./target/release/agent-rs \
    --task "Analyze this text and predict what follows..." \
    --temperature 0.0 \
    --timeout 30000 \
    --max-steps 5
```

### Command Line Options

- `--task <TASK>`: Task description for the agent to complete (required)
- `--mcp-config <PATH>`: Path to MCP configuration JSON file (default: "mcp_config.json")
- `--max-steps <STEPS>`: Maximum reasoning/tool steps for the agent (default: 7)
- `--timeout <MS>`: Timeout in milliseconds (default: 300000)
- `--temperature <TEMP>`: Temperature for the model - should be 0.0 for determinism (default: 0.0)

### Environment Variables

The agent requires a Gemini API key:

```bash
export GEMINI_API_KEY="your-api-key-here"
# OR
export GOOGLE_API_KEY="your-api-key-here"
```

### Output Format

The agent outputs results in a format compatible with the existing Python agent:

```
AGENT_RESULT_JSON:{"final_text": "predicted text content", "duration_ms": 1234}
predicted text content
```

## Determinism Validation

The agent includes comprehensive tests to validate deterministic behavior:

```bash
# Run all tests (requires API key)
cargo test

# Run specific determinism test
cargo test test_deterministic_output
```

### Test Requirements

For the determinism tests to run, you need:
1. A valid `GEMINI_API_KEY` or `GOOGLE_API_KEY` environment variable
2. Internet connection to access the Gemini API

If no API key is available, tests will be skipped with a message.

## Integration with CandleZip

This agent can be used as a drop-in replacement for the Python agent in the compression pipeline:

1. **Build the agent**: `cargo build --release`
2. **Test determinism**: `cargo test` (with API key set)
3. **Integration**: Replace Python agent calls with calls to `./agent-rs/target/release/agent-rs`

The command-line interface and output format are designed to be compatible with the existing compression system.

## Architecture

### Key Components

1. **Direct Gemini API Client**: Custom HTTP client for Gemini API
2. **Deterministic Configuration**: Enforced temperature=0, seed=42
3. **Request/Response Structures**: Strongly-typed API communication
4. **Environment Setup**: Deterministic runtime environment variables
5. **Comprehensive Testing**: Multi-run determinism validation

### Why Direct API vs Rig Framework?

Initially, this implementation attempted to use the Rig framework, but encountered compatibility issues with `let_chains` syntax requiring newer Rust features. The direct API approach provides:

- Better compatibility across Rust versions
- More control over request parameters
- Simpler dependencies
- Easier to audit for determinism

## Future Enhancements

1. **MCP Tool Integration**: Support for Model Context Protocol tools
2. **Streaming Responses**: For real-time processing
3. **Response Caching**: Local caching for improved performance
4. **Multi-model Support**: Support for other deterministic models

## Testing Determinism

The agent includes several test categories:

1. **Deterministic Output Test**: Verifies same input → same output
2. **Different Input Test**: Verifies different inputs → different outputs  
3. **Configuration Validation**: Ensures proper default settings
4. **MCP Config Loading**: Tests configuration file parsing

## Troubleshooting

### Common Issues

1. **Missing API Key**: Set `GEMINI_API_KEY` environment variable
2. **Network Timeouts**: Check internet connection and increase timeout
3. **API Rate Limits**: Gemini API has rate limits; add delays between calls
4. **Build Errors**: Ensure you have a recent Rust toolchain (1.70+)

### Debug Mode

Run with debug logging:

```bash
RUST_LOG=debug ./target/release/agent-rs --task "your task"
```

## License

This implementation is part of the CandleZip compression research project.
