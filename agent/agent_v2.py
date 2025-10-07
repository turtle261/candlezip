"""CrewAI Agent CLI with MCP tools and Gemini integration.

A clean implementation of a CrewAI agent with:
- Gemini 2.5 Flash as the default LLM
- MCP tools integration from mcp_config.json
- Agent memory with Gemini embeddings
- Reasoning and planning capabilities

Usage:
  python agent_cli.py --task "Your task description here"
"""

import random
random.seed(42)
import argparse
import json
import os
import sys
from typing import List, Optional, Dict, Any

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv, find_dotenv
from mcp import StdioServerParameters
import time
from crewai_tools import WebsiteSearchTool

# Example of initiating tool that agents can use 
# to search across any discovered websites
# * Enforce deterministic runtime environment
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("TZ", "UTC")
os.environ.setdefault("LANG", "C")
os.environ.setdefault("LC_ALL", "C")
os.environ.setdefault("CREWAI_TELEMETRY_DISABLED", "1")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _now_ms() -> int:
    return int(time.time() * 1000)


def setup_file_specific_memory(input_file_path: str) -> str:
    """Setup file-specific memory storage to prevent cross-file contamination.
    
    Args:
        input_file_path: Path to the input file being compressed
        
    Returns:
        Memory collection identifier for this file
    """
    # Create a deterministic but unique identifier for this input file
    import hashlib
    file_id = hashlib.md5(os.path.abspath(input_file_path).encode()).hexdigest()[:8]
    file_basename = os.path.splitext(os.path.basename(input_file_path))[0]
    
    # Combine for readable yet unique collection name
    collection_name = f"candlezip_{file_basename}_{file_id}"
    
    # Set environment variables for CrewAI memory
    os.environ["CREWAI_MEMORY_COLLECTION"] = collection_name
    
    return collection_name


def create_learning_callback(chunk_index: int, input_file_path: str) -> callable:
    """Create a task callback that records learning information based on compression results.
    
    Args:
        chunk_index: The current chunk being processed
        input_file_path: Path to the input file being compressed
        
    Returns:
        Callback function for task completion
    """
    def learning_callback(task_output):
        """Record learning information about this agent execution for future improvement."""
        try:
            # Get gating result from environment variables set by Rust
            gate_result = os.environ.get("CANDLEZIP_LAST_GATE", "0")
            bits_saved = float(os.environ.get("CANDLEZIP_LAST_BITS_SAVED", "0.0"))
            baseline_bits = float(os.environ.get("CANDLEZIP_LAST_BASELINE_BITS", "0.0"))
            candidate_id = int(os.environ.get("CANDLEZIP_LAST_CANDIDATE_ID", "0"))
            budget_id = int(os.environ.get("CANDLEZIP_LAST_BUDGET_ID", "0"))
            
            # Create learning record
            learning_entry = {
                "type": "compression_learning",
                "chunk_index": chunk_index,
                "file_path": input_file_path,
                "gate_result": int(gate_result),
                "bits_saved": bits_saved,
                "baseline_bits": baseline_bits,
                "candidate_id": candidate_id,
                "budget_id": budget_id,
                "agent_output": str(task_output.raw),
                "success": gate_result == "1",
                "timestamp": time.time()
            }
            
            if gate_result == "1":
                # Success - record what worked
                success_summary = (
                    f"COMPRESSION SUCCESS (Chunk {chunk_index}): Your prediction strategy succeeded! "
                    f"You saved {bits_saved:.2f} bits (baseline: {baseline_bits:.2f} bits). "
                    f"The successful approach used candidate {candidate_id} with budget {budget_id}. "
                    f"Key success factors: Your output provided precise contextual information that reduced prediction uncertainty. "
                    f"Remember this pattern and strategy for similar text content in future chunks."
                )
            else:
                # Failure - record what didn't work  
                success_summary = (
                    f"COMPRESSION ATTEMPT (Chunk {chunk_index}): Your prediction did not improve compression "
                    f"(saved {bits_saved:.2f} bits from baseline {baseline_bits:.2f}). "
                    f"The attempted approach used candidate {candidate_id} with budget {budget_id}. "
                    f"Learning opportunity: Consider different prediction strategies, more specific content, "
                    f"or alternative tool usage patterns for this type of text content."
                )
            
            # Store in agent's long-term memory through a simulated task execution
            # CrewAI will automatically store this in long-term memory for future retrieval
            learning_entry["summary"] = success_summary
            
            # Save learning data to a JSON file that can be read by future agent runs
            memory_dir = os.environ.get("CREWAI_STORAGE_DIR", os.path.join(os.getcwd(), "agent_memory"))
            os.makedirs(memory_dir, exist_ok=True)
            collection_name = os.environ.get("CREWAI_MEMORY_COLLECTION", "default")
            learning_file = os.path.join(memory_dir, f"{collection_name}_learning.jsonl")
            
            with open(learning_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(learning_entry) + "\n")
                
            print(f"[Learning] Recorded result for chunk {chunk_index}: {'SUCCESS' if gate_result == '1' else 'ATTEMPT'}")
            
        except Exception as e:
            print(f"[Learning] Warning: Failed to record learning data: {e}")
            
    return learning_callback


def load_learning_context(input_file_path: str, current_chunk: int) -> str:
    """Load relevant learning context from previous chunks for this file.
    
    Args:
        input_file_path: Path to the input file being compressed
        current_chunk: Current chunk index
        
    Returns:
        Learning context string to add to task description
    """
    try:
        memory_dir = os.environ.get("CREWAI_STORAGE_DIR", os.path.join(os.getcwd(), "agent_memory"))
        collection_name = os.environ.get("CREWAI_MEMORY_COLLECTION", "default")
        learning_file = os.path.join(memory_dir, f"{collection_name}_learning.jsonl")
        
        if not os.path.exists(learning_file):
            return ""
            
        # Load recent learning entries (last 10 chunks or successful ones)
        recent_learning = []
        successful_strategies = []
        
        with open(learning_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    if entry.get("chunk_index", 0) < current_chunk:  # Only past chunks
                        recent_learning.append(entry)
                        if entry.get("success", False):
                            successful_strategies.append(entry)
                except:
                    continue
        
        if not recent_learning:
            return ""
            
        # Sort by chunk index and take most recent
        recent_learning.sort(key=lambda x: x.get("chunk_index", 0))
        recent_entries = recent_learning[-10:]  # Last 10 chunks
        
        context_parts = ["\n\n[LEARNING CONTEXT FROM PREVIOUS CHUNKS]"]
        
        if successful_strategies:
            success_count = len(successful_strategies)
            total_count = len(recent_learning)
            context_parts.append(f"Success Rate: {success_count}/{total_count} chunks improved compression")
            
            # Add summary of successful strategies
            context_parts.append("\nSUCCESSFUL STRATEGIES:")
            for entry in successful_strategies[-3:]:  # Last 3 successes
                chunk_idx = entry.get("chunk_index", 0)
                bits_saved = entry.get("bits_saved", 0)
                candidate_id = entry.get("candidate_id", 0)
                context_parts.append(f"- Chunk {chunk_idx}: Saved {bits_saved:.2f} bits using candidate {candidate_id}")
        
        # Add recent attempts context  
        if recent_entries:
            context_parts.append(f"\nRECENT ATTEMPTS (Chunks {recent_entries[0].get('chunk_index', 0)}-{recent_entries[-1].get('chunk_index', 0)}):")
            success_recent = sum(1 for e in recent_entries if e.get("success", False))
            context_parts.append(f"Recent success rate: {success_recent}/{len(recent_entries)}")
        
        context_parts.append("\nREMEMBER: Successful predictions provide specific, information-dense content that reduces token prediction uncertainty. Failed attempts often lack specificity or don't match actual upcoming content patterns.\n")
        
        return "\n".join(context_parts)
        
    except Exception as e:
        print(f"[Learning] Warning: Failed to load learning context: {e}")
        return ""


def load_mcp_servers(config_path: str = "mcp_config.json") -> List[StdioServerParameters]:
    """Load MCP server configurations from JSON file.
    
    Args:
        config_path: Path to MCP configuration file
        
    Returns:
        List of MCP server parameters
    """
    if not os.path.exists(config_path):
        return []
    
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        
        servers = []
        for name, spec in config.get("mcpServers", {}).items():
            if command := spec.get("command"):
                servers.append(StdioServerParameters(
                    command=command,
                    args=spec.get("args", []),
                    env={**os.environ, **spec.get("env", {})}
                ))
        return servers
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Failed to load MCP config: {e}")
        return []

def build_embedder_config() -> dict: # remove for nomem
    """Build Google embedder config after environment is loaded.
    
    Returns:
        Dict[str, Any]: Provider configuration for CrewAI embedder.
    """
    # Prefer Google AI Studio key for embeddings, fall back to GEMINI_API_KEY for compatibility
    google_api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    # Default to latest Google embedding model naming
    model_name = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
    return {
        "provider": "google-generativeai",
        "config": {
            "model": model_name,
            "task_type": "retrieval_document",
            "api_key_env_var": "GOOGLE_API_KEY",
            "api_key": google_api_key,
        },
    }
 

def create_agent(task_description: str, max_steps: int, embedder_config: dict) -> Agent:
    """Create a CrewAI agent with Gemini LLM and memory.
    
    Args:
        task_description: The task the agent will perform
        max_steps: Maximum reasoning attempts
        embedder_config: Configuration for embeddings used by memory system
        
    Returns:
        Configured CrewAI agent with long-term memory enabled
    """
    # Configure Gemini LLM
    import json

    file_path = 'vertcred.json'

    # Load the JSON file
    with open(file_path, 'r') as file:
        vertex_credentials = json.load(file)

    # Convert the credentials to a JSON string
    vertex_credentials_json = json.dumps(vertex_credentials)
    gEmbed = build_embedder_config()
        #For regular:
    llm = LLM(
        model="gemini/gemini-2.5-flash", # Do not change this.
        include_reasoning=True,
        reasoning_effort="high",
        temperature=0.0,
        seed=42,
        top_p=0.95,
        #vertex_credentials=vertex_credentials_json

    )
    # For reasoning:
    llm2 = LLM(
        model="gemini/gemini-2.5-pro", # Do not change this.
        temperature=0.0,
        include_reasoning=True,
        reasoning_effort="medium",
        seed=42,
        top_p=0.95,
        #vertex_credentials=vertex_credentials_json
    )
    llm3=llm
    return Agent(
        role="Predictive Compression Intelligence",
        goal="Minimize future token prediction entropy by providing maximally informative context through strategic tool usage and learned patterns",
        backstory=(
            "You are a specialized intelligence system optimizing predictive compression under the Sink-Inclusive Description Length (SIMDL) framework. "
            "Your objective is to reduce information entropy for upcoming token predictions by accessing external information sources (entropy sinks) at measured cost. "
            "\n\nCore Competencies:"
            "\n- ENTROPY MINIMIZATION: You understand that your predictions directly reduce the bits required to encode future tokens - successful hints lower the model's uncertainty about what comes next"
            "\n- TOOL-AUGMENTED REASONING: You leverage available external knowledge sources dynamically to retrieve exact, information-dense content that aids prediction"
            "\n- ADAPTIVE LEARNING: You maintain memory of which prediction strategies succeed or fail, continuously refining your approach based on empirical compression gains"
            "\n- PATTERN SYNTHESIS: You identify salient features in text prefixes (domains, entities, structural markers) that signal retrievable content or predictable continuations"
            "\n\nOperational Strategy:"
            "\n- Analyze the provided text prefix to infer domain, authorship, structural patterns, and likely source material"
            "\n- Query external knowledge sources to locate exact matching content or highly relevant contextual material, or predictive information."
            "\n- When exact sources are identified, extract verbatim continuations; otherwise synthesize high-confidence predictions from available context"
            "\n- Prioritize information-dense elements: technical terminology, proper nouns, numerical data, domain-specific jargon, and structural markers"
            "\n- Include natural surrounding text to preserve authentic linguistic flow and token-level predictability"
            "\n\nPerformance Criterion: Your output is evaluated by measuring cross-entropy reduction over future token predictions. "
            "Successful hints provide precise, context-rich information that allows the language model to assign higher probability mass to the actual upcoming tokens, "
            "thereby reducing the expected description length under arithmetic coding. You understand that ALL tools that you currently have access to are counted as external sources, and that you should use any tools at your disposal to do tasks. You will need to think very hard about how your toolset may be able to get the information you need, and get that information efficiently. YOU ONLY have access to this directory through your tools: /C/Users/Noah/Documents/sink ONLY. Start by analyzing the directory and then use the tools correctly to solve the task."
        ),
        llm=llm,
        memory=False,
        reasoning=True,
        planning=True,
        max_reasoning_attempts=2,
        max_planning_attempts=2,
        verbose=True,
        allow_delegation=False,
        max_iter=2*max_steps,
        reasoning_llm=llm2,
        planning_llm=llm2,
        function_calling_llm=llm3,
        max_rpm=30,
        max_retry_attempts=2,
    )


def run_agent(task_description: str, mcp_config_path: str, max_steps: int) -> str:
    """Run the agent with the given task and long-term memory enabled.
    
    Args:
        task_description: Description of the task to perform
        mcp_config_path: Path to MCP configuration file
        max_steps: Maximum reasoning steps
        
    Returns:
        Agent's response
    """
    # Get file-specific parameters from environment
    input_file_path = os.environ.get("CANDLEZIP_INPUT_FILE", "unknown_file")
    chunk_index = int(os.environ.get("CANDLEZIP_CHUNK_INDEX", "0"))
    
    # Setup file-specific memory to prevent cross-file contamination
    collection_name = setup_file_specific_memory(input_file_path)
    print(f"[Memory] Using collection: {collection_name}")
    
    # Disable embedder/memory entirely in no-memory agent
    embedder_config = None
    
    # Set up CrewAI storage directory to be inside the run's scan directory (same dir as proof.csv)
    # Respect CANDLEZIP_WATCHDOG_DIR when present; otherwise fall back to local ./agent_memory
    run_dir = os.environ.get("CANDLEZIP_WATCHDOG_DIR")
    storage_dir = os.path.abspath(os.path.join(run_dir if run_dir else os.getcwd(), "agent_memory"))
    os.environ["CREWAI_STORAGE_DIR"] = storage_dir
    # Enable persistence for long-term learning
    os.environ.pop("CREWAI_DISABLE_PERSISTENCE", None)  # Remove disable flag if present
    
    # Create agent with memory enabled
    agent = create_agent(task_description, max_steps, embedder_config)
    
    # No cross-chunk memory: use task description only
    enhanced_task_description = task_description
    
    # Disable learning callback: learning entries are written from Rust after gating to ensure exact alignment
    task_callback = None
    
    # Force plain text continuation only; avoid meta chatter
    task = Task(
        description=enhanced_task_description,
        expected_output="Plain text continuation only. No markdown, no analysis, no headings.",
        agent=agent
    )
    
    # Load MCP tools
    mcp_servers = load_mcp_servers(mcp_config_path)
    
    # Try to run with MCP tools if available
    if mcp_servers:
        try:
            with MCPServerAdapter(mcp_servers) as mcp_tools:
                # Convert to list and check if we have any tools
                tools_list = list(mcp_tools) if mcp_tools else []
                if tools_list:
                    agent.tools = tools_list
                    print(f"[MCP] Loaded {len(tools_list)} tools: {[tool.name for tool in tools_list]}")
                else:
                    print("[MCP] Warning: No tools available from MCP servers")
                
                #webtool = WebsiteSearchTool(embedder=embedder_config)
                #gent.tools.append(webtool)
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                    memory=False,
                    embedder=None,
                    task_callback=task_callback  # Add learning callback
                )
                t0 = _now_ms()
                result = crew.kickoff()
                t1 = _now_ms()
                final_text = str(result.raw)
                print(f"AGENT_RESULT_JSON:{{\"final_text\": {json.dumps(final_text)}, \"duration_ms\": {t1 - t0}}}")
                return final_text
        except Exception as e:
            print(f"Warning: Running without MCP due to configuration/runtime error: {e}")
            # Fall through to run without MCP tools
    
    # Run without MCP tools, no memory
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True,
        memory=False,
        task_callback=task_callback  # Add learning callback
    )
    t0 = _now_ms()
    result = crew.kickoff()
    t1 = _now_ms()
    final_text = str(result.raw)
    print(f"AGENT_RESULT_JSON:{{\"final_text\": {json.dumps(final_text)}, \"duration_ms\": {t1 - t0}}}")
    return final_text


def main() -> int:
    """Main entry point."""
    # Load environment variables
    env_path = find_dotenv()
    load_dotenv(env_path if env_path else None)
    
    # Set random seed for determinism
    random.seed(42)
    
    # Check for required API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is required", file=sys.stderr)
        return 1
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CrewAI Agent CLI with MCP tools and Gemini integration"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task",
        help="Task description for the agent to complete"
    )
    group.add_argument(
        "--task-file",
        dest="task_file",
        help="Path to a file containing the task description"
    )
    parser.add_argument(
        "--mcp-config",
        default="mcp_config.json",
        help="Path to MCP configuration JSON file"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=15,
        help="Maximum reasoning/tool steps for the agent"
    )
    
    args = parser.parse_args()

    # Resolve task text (allow long text via file to avoid Windows cmd length limits)
    task_text: str
    if getattr(args, "task_file", None):
        with open(args.task_file, "r", encoding="utf-8") as f:
            task_text = f.read()
    else:
        task_text = args.task
    
    try:
        result = run_agent(task_text, args.mcp_config, args.max_steps)
        print(result)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

