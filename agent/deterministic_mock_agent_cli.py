"""
Mock Agent CLI - Deterministic placeholder for debug/testing.

Always returns the same hardcoded response to match the attached example.
Can be used as a 1:1 drop-in replacement for agent_cli.py.
"""

import argparse
import json
import sys
import time


def main() -> int:
    """Mock main entry point that always returns the same response."""

    # Parse arguments (same as real agent_cli.py for compatibility)
    parser = argparse.ArgumentParser(
        description="Mock CrewAI Agent CLI with MCP tools and Gemini integration"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Task description for the agent to complete"
    )
    parser.add_argument(
        "--mcp-config",
        default="mcp_config.json",
        help="Path to MCP configuration JSON file (ignored in mock)"
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=7,
        help="Maximum reasoning/tool steps for the agent (ignored in mock)"
    )

    args = parser.parse_args()

    # Simulate processing time (matches the example duration)
    time.sleep(0.1)  # Brief pause to simulate work

    # Hardcoded response from the attached example
    final_text = (
        "The document will likely detail the methodology for estimating the "
        "asymptotic upper bound of English entropy using the LLaMA-7B large "
        "language model as a predictor, yielding significantly lower estimates "
        "(e.g., 0.692 bits/character on text8 dataset) compared to prior work "
        "(e.g., 1.75 bits/character from word trigram models). It will present "
        "the LLMZip algorithm, which combines LLM predictions with arithmetic "
        "coding for lossless text compression. Experimental results will be "
        "presented, demonstrating LLMZip's superior performance over state-of-the-art "
        "schemes like BSC, ZPAQ, and paq8h, with specific compression ratios reported "
        "(e.g., LLMZip achieving 0.71 bits/character on text8, outperforming ZPAQ "
        "at 1.4 bits/char and paq8h at 1.2 bits/char). The paper will likely "
        "elaborate on the theoretical connections between prediction and compression, "
        "referencing foundational work by Shannon and prior research on language "
        "models with arithmetic coding."
    )

    # Output JSON result (matches real agent's format)
    result_json = {
        "final_text": final_text,
        "duration_ms": 28188
    }
    print(f"AGENT_RESULT_JSON:{json.dumps(result_json)}")

    # Print the final text
    print(final_text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
