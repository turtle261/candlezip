"""CrewAI Agent CLI with MCP tools and Gemini integration.

A clean implementation of a CrewAI agent with:
- Gemini 2.5 Flash as the default LLM
- MCP tools integration from mcp_config.json
- Agent memory with Gemini embeddings
- Reasoning and planning capabilities

Usage:
  python agent_cli.py --task "Your task description here"
"""

import argparse
import json
import os
import sys
from typing import List, Optional

from crewai import Agent, Crew, Process, Task, LLM
from crewai_tools import MCPServerAdapter
from dotenv import load_dotenv, find_dotenv
from mcp import StdioServerParameters
import time


def _now_ms() -> int:
    return int(time.time() * 1000)


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


def create_agent(task_description: str, max_steps: int) -> Agent:
    """Create a CrewAI agent with Gemini LLM and memory.
    
    Args:
        task_description: The task the agent will perform
        
    Returns:
        Configured CrewAI agent
    """
    # Configure Gemini LLM
    llm = LLM(
        model="gemini/gemini-2.5-flash", # Do not change this.
        temperature=0.0,
    )
    llm2 = LLM(
        model="gemini/gemini-2.5-flash", # Do not change this.
        temperature=0.0,
    )
    # Configure embedder for agent memory
    embedder_config = {
        "provider": "google",
        "config": {
            "model": "gemini-embedding-001",
            "task_type": "retrieval_document"
        }
    }
    
    return Agent(
        role="AI Assistant",
        goal="Complete the given task accurately and efficiently using available tools",
        backstory=(
            "You are a capable AI assistant with access to various tools through MCP. "
            "You approach tasks methodically, use tools when appropriate, and provide "
            "clear, accurate responses."
        ),
        llm=llm,
        memory=True,
        embedder=embedder_config,
        reasoning=True,
        max_reasoning_attempts=max_steps,
        verbose=True,
        allow_delegation=False,
        max_iter=max(10, max_steps + 3),
        reasoning_llm=llm2,
        max_rpm=7,
    )


def run_agent(task_description: str, mcp_config_path: str, max_steps: int) -> str:
    """Run the agent with the given task.
    
    Args:
        task_description: Description of the task to perform
        
    Returns:
        Agent's response
    """
    # Load MCP tools
    mcp_servers = load_mcp_servers(mcp_config_path)
    
    # Create agent
    agent = create_agent(task_description, max_steps)
    
    # Create task
    task = Task(
        description=task_description,
        expected_output="A comprehensive and accurate response to the task",
        agent=agent
    )
    
    # Try to run with MCP tools if available
    if mcp_servers:
        try:
            with MCPServerAdapter(mcp_servers) as mcp_tools:
                agent.tools = list(mcp_tools)
                crew = Crew(
                    agents=[agent],
                    tasks=[task],
                    process=Process.sequential,
                    verbose=True,
                )
                t0 = _now_ms()
                result = crew.kickoff()
                t1 = _now_ms()
                final_text = str(result.raw)
                print(f"AGENT_RESULT_JSON:{{\"final_text\": {json.dumps(final_text)}, \"duration_ms\": {t1 - t0}}}")
                return final_text
        except Exception as e:
            print(f"Warning: MCP tools failed to load: {e}")
            # Fall through to run without MCP tools
    
    # Run without MCP tools
    crew = Crew(
        agents=[agent],
        tasks=[task],
        process=Process.sequential,
        verbose=True
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
    
    # Check for required API key
    if not os.getenv("GEMINI_API_KEY"):
        print("Error: GEMINI_API_KEY environment variable is required", file=sys.stderr)
        return 1
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="CrewAI Agent CLI with MCP tools and Gemini integration"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task description for the agent to complete"
    )
    parser.add_argument(
        "--mcp-config",
        default="mcp_config.json",
        help="Path to MCP configuration JSON file"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=7,
        help="Maximum reasoning/tool steps for the agent"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_agent(args.task, args.mcp_config, args.max_steps)
        print(result)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())


