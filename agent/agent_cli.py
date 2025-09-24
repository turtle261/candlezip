#!/usr/bin/env python3
# * Wrapper CLI that delegates to the Rust `agent-rs` binary.
# * Accepts the same CLI args as `agent/agent_cli.py` and forwards them.
# * Prints the wrapped tool's stdout/stderr and exits with its return code.

import argparse
import os
import platform
import shutil
import subprocess
import sys


def find_agent_rs_binary() -> str:
    """Find the built `agent-rs` binary in the repository.

    Returns the path to the executable. Raises FileNotFoundError if not found.
    """
    # Prefer an explicit absolute path on Windows per user request
    # Compute repo root as parent of the `agent/` directory (this file's directory)
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    preferred_windows = r"C:\Users\Noah\Documents\business\resumeportfolio\candlezip\agent-rs\target\release\agent-rs.exe"
    candidate_windows = os.path.join(repo_root, "agent-rs", "target", "release", "agent-rs.exe")
    candidate_unix = os.path.join(repo_root, "agent-rs", "target", "release", "agent-rs")

    if platform.system() == "Windows":
        if os.path.exists(preferred_windows):
            return preferred_windows
        if os.path.exists(candidate_windows):
            return candidate_windows
    else:
        if os.path.exists(candidate_unix):
            return candidate_unix

    # Fall back to looking on PATH for `agent-rs`
    bin_on_path = shutil.which("agent-rs")
    if bin_on_path:
        return bin_on_path

    raise FileNotFoundError("agent-rs binary not found. Build the Rust binary or add it to PATH.")


def build_subprocess_args(parsed_args: argparse.Namespace) -> list:
    """Construct the subprocess command-line to invoke agent-rs with equivalent args."""
    cmd = [find_agent_rs_binary(), "--task", parsed_args.task]

    # Forward optional args only if provided (keeps parity with original CLI)
    if parsed_args.mcp_config:
        cmd += ["--mcp-config", parsed_args.mcp_config]
    if parsed_args.max_steps is not None:
        cmd += ["--max-steps", str(parsed_args.max_steps)]
    if parsed_args.timeout is not None:
        cmd += ["--timeout", str(parsed_args.timeout)]
    if parsed_args.temperature is not None:
        cmd += ["--temperature", str(parsed_args.temperature)]
    if parsed_args.model:
        cmd += ["--model", str(parsed_args.model)]

    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description="Wrapper CLI that calls the Rust agent-rs binary")
    parser.add_argument("--task", required=True, help="Task description for the agent to complete")
    parser.add_argument("--mcp-config", default="mcp_config.json", help="Path to MCP configuration JSON file")
    parser.add_argument("--max-steps", type=int, default=7, help="Maximum reasoning/tool steps for the agent")
    parser.add_argument("--timeout", type=int, default=300000, help="Timeout in milliseconds")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature for the model (0.0 for determinism)")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Gemini model name for deterministic runs")

    args = parser.parse_args()

    # Ensure GEMINI_API_KEY / GOOGLE_API_KEY presence like agent_cli.py
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        print("Error: GEMINI_API_KEY environment variable is required", file=sys.stderr)
        return 1

    try:
        cmd = build_subprocess_args(args)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Forward the current environment
    env = os.environ.copy()

    # Run the binary and stream output
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            text=True,
            bufsize=1,
        )

        # Capture stdout and stderr while streaming
        stdout_lines = []
        stderr_lines = []

        # Read until process exits
        out, err = proc.communicate()
        if out:
            print(out, end="")
            stdout_lines = out.splitlines()
        if err:
            print(err, file=sys.stderr, end="")
            stderr_lines = err.splitlines()

        return proc.returncode or 0

    except KeyboardInterrupt:
        try:
            proc.terminate()
        except Exception:
            pass
        print("Interrupted", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Failed to run agent-rs: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
