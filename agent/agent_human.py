"""Human-in-the-Loop Agent for CandleZip Compression.

A research-grade web GUI that allows humans to perform the compression prediction task,
with full access to MCP tools in an intuitive interface. Maintains API compatibility
with the LLM agent for seamless integration with CandleZip.

Usage:
  python agent_human.py --task "Your task description here" --mcp-config mcp_config.json
"""

import argparse
import json
import os
import sys
import time
import threading
import webbrowser
from pathlib import Path
from typing import List, Dict, Any, Optional

from flask import Flask, render_template_string, request, jsonify
from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters
from dotenv import load_dotenv, find_dotenv


# ============================================================================
# MCP Tool Manager
# ============================================================================

class MCPToolManager:
    """Manages MCP tool servers and provides tool invocation interface."""
    
    def __init__(self, mcp_config_path: str):
        self.mcp_config_path = mcp_config_path
        self.servers = []
        self.tools = []
        self.adapter = None            # MCPServerAdapter instance (context manager)
        self.adapter_ctx = None        # Value returned by __enter__, iterable over tools
        
    def load_tools(self):
        """Load MCP servers and extract available tools."""
        if not os.path.exists(self.mcp_config_path):
            print(f"Warning: MCP config not found at {self.mcp_config_path}", file=sys.stderr)
            return
        
        try:
            with open(self.mcp_config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            
            servers = []
            for name, spec in config.get("mcpServers", {}).items():
                if command := spec.get("command"):
                    servers.append(StdioServerParameters(
                        command=command,
                        args=spec.get("args", []),
                        env={**os.environ, **spec.get("env", {})}
                    ))
            
            self.servers = servers
            
            # Initialize MCP adapter and extract tools (keep context open for UI lifetime)
            if servers:
                self.adapter = MCPServerAdapter(servers)
                self.adapter_ctx = self.adapter.__enter__()
                try:
                    self.tools = list(self.adapter_ctx) if self.adapter_ctx else []
                except TypeError:
                    # Fallback: some versions expose .tools attribute
                    self.tools = list(getattr(self.adapter_ctx, "tools", []))
                print(f"[MCP] Loaded {len(self.tools)} tools: {[getattr(t,'name',str(t)) for t in self.tools]}", file=sys.stderr)
            
        except Exception as e:
            print(f"Warning: Failed to load MCP tools: {e}", file=sys.stderr)
    
    def get_tool_list(self) -> List[Dict[str, Any]]:
        """Get list of available tools with metadata."""
        tool_list = []
        for tool in self.tools:
            tool_info = {
                "name": tool.name,
                "description": getattr(tool, "description", "No description available"),
                "args_schema": self._extract_args_schema(tool)
            }
            tool_list.append(tool_info)
        return tool_list
    
    def _extract_args_schema(self, tool) -> Dict[str, Any]:
        """Extract argument schema from tool."""
        try:
            # Try to get args_schema if available
            if hasattr(tool, "args_schema"):
                schema = tool.args_schema
                if hasattr(schema, "schema"):
                    return schema.schema()
                elif isinstance(schema, dict):
                    return schema
            
            # Try to inspect the tool's run method
            if hasattr(tool, "run"):
                import inspect
                sig = inspect.signature(tool.run)
                params = {}
                for name, param in sig.parameters.items():
                    if name not in ["self", "args", "kwargs"]:
                        params[name] = {
                            "type": "string",
                            "description": f"Parameter: {name}"
                        }
                return {"properties": params, "type": "object"}
        except Exception as e:
            print(f"Warning: Could not extract schema for {tool.name}: {e}", file=sys.stderr)
        
        return {"properties": {}, "type": "object"}
    
    def invoke_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Invoke a tool with given arguments."""
        try:
            # Find the tool
            tool = None
            for t in self.tools:
                if t.name == tool_name:
                    tool = t
                    break
            
            if tool is None:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found"
                }
            
            # Invoke the tool
            print(f"[MCP] Invoking tool: {tool_name} with args: {args}", file=sys.stderr)
            
            # Different tools may accept arguments differently
            result = None
            call_errors = []
            # 1) If tool is directly callable (__call__)
            try:
                if callable(tool):
                    result = tool(**args)
                else:
                    raise TypeError("tool not callable")
            except Exception as e:
                call_errors.append(f"callable: {e}")
            # 2) Try .run(**kwargs)
            if result is None:
                try:
                    if hasattr(tool, 'run'):
                        result = tool.run(**args)
                except Exception as e:
                    call_errors.append(f"run_kwargs: {e}")
            # 3) Try .run(args)
            if result is None:
                try:
                    if hasattr(tool, 'run'):
                        result = tool.run(args)
                except Exception as e:
                    call_errors.append(f"run_dict: {e}")
            # 4) Try positional run(*values)
            if result is None:
                try:
                    if hasattr(tool, 'run'):
                        result = tool.run(*args.values())
                except Exception as e:
                    call_errors.append(f"run_pos: {e}")
            # If still None, raise last error
            if result is None and call_errors:
                raise RuntimeError("; ".join(call_errors))
            
            # Normalize result to string or JSON-friendly
            try:
                # If it's already a string-like
                if isinstance(result, (str, bytes)):
                    out = result.decode('utf-8', errors='ignore') if isinstance(result, bytes) else result
                else:
                    # Try JSON serialize
                    out = json.dumps(result, default=lambda o: getattr(o, '__dict__', str(o)), ensure_ascii=False, indent=2)
            except Exception:
                out = str(result)

            return {"success": True, "result": out}
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"[MCP] Tool invocation error: {error_details}", file=sys.stderr)
            return {
                "success": False,
                "error": str(e),
                "details": error_details
            }
    
    def cleanup(self):
        """Clean up MCP adapter."""
        if self.adapter:
            try:
                self.adapter.__exit__(None, None, None)
            except Exception as e:
                print(f"Warning: Error cleaning up MCP adapter: {e}", file=sys.stderr)


# ============================================================================
# Flask Web Application
# ============================================================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'candlezip-hitl-secret-key'
app.config['SERVER_HOST'] = 'localhost'
app.config['SERVER_PORT'] = 5000

# Global state
task_data = {
    "task": "",
    "prefix": "",
    "prior_memory": "",
    "start_time": 0,
    "final_text": None,
    "tool_history": [],
    "completed": False
}

tool_manager = None
printed_result = False  # ensure we don't emit AGENT_RESULT_JSON twice


# HTML Template with modern, research-grade UI
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CandleZip HITL - Human Intelligence Compression</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
            color: #333;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: 600;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.95;
        }
        
        .timer {
            background: rgba(255, 255, 255, 0.2);
            display: inline-block;
            padding: 10px 20px;
            border-radius: 25px;
            margin-top: 15px;
            font-size: 1.2em;
            font-weight: bold;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 30px;
        }
        
        .panel {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 25px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
        }
        
        .panel h2 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.5em;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }
        
        .task-section {
            grid-column: 1 / -1;
        }
        
        .text-display {
            background: white;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            max-height: 300px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 0.9em;
            line-height: 1.6;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        
        .tool-list {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .tool-item {
            background: white;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .tool-item:hover {
            border-color: #667eea;
            box-shadow: 0 4px 12px rgba(102, 126, 234, 0.2);
            transform: translateY(-2px);
        }
        
        .tool-item.active {
            border-color: #667eea;
            background: #f0f4ff;
        }
        
        .tool-name {
            font-weight: 600;
            color: #667eea;
            font-size: 1.1em;
            margin-bottom: 5px;
        }
        
        .tool-description {
            color: #666;
            font-size: 0.9em;
        }
        
        .tool-form {
            background: white;
            border: 2px solid #667eea;
            border-radius: 8px;
            padding: 20px;
            margin-top: 15px;
            display: none;
        }
        
        .tool-form.active {
            display: block;
        }
        
        .form-group {
            margin-bottom: 15px;
        }
        
        .form-group label {
            display: block;
            font-weight: 600;
            margin-bottom: 5px;
            color: #333;
        }
        
        .form-group input,
        .form-group textarea {
            width: 100%;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 0.95em;
            font-family: inherit;
        }
        
        .form-group textarea {
            min-height: 80px;
            resize: vertical;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            box-shadow: none;
        }
        
        .btn-secondary {
            background: #6c757d;
            box-shadow: 0 4px 15px rgba(108, 117, 125, 0.4);
        }
        
        .tool-history {
            max-height: 400px;
            overflow-y: auto;
        }
        
        .history-item {
            background: white;
            border-left: 4px solid #667eea;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 12px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        }
        
        .history-header {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 8px;
        }
        
        .history-result {
            background: #f8f9fa;
            border-radius: 4px;
            padding: 10px;
            font-family: 'Courier New', monospace;
            font-size: 0.85em;
            white-space: pre-wrap;
            word-wrap: break-word;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .prediction-section {
            grid-column: 1 / -1;
            background: #fff3cd;
            border: 2px solid #ffc107;
        }
        
        .prediction-section h2 {
            color: #ff6b6b;
            border-bottom-color: #ff6b6b;
        }
        
        #finalPrediction {
            width: 100%;
            min-height: 150px;
            padding: 15px;
            border: 2px solid #ffc107;
            border-radius: 8px;
            font-size: 1em;
            font-family: 'Courier New', monospace;
            resize: vertical;
        }
        
        .submit-btn {
            background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
            box-shadow: 0 4px 15px rgba(255, 107, 107, 0.4);
            font-size: 1.2em;
            padding: 15px 40px;
            margin-top: 15px;
        }
        
        .submit-btn:hover {
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.6);
        }
        
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        
        .warning-box {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        
        .success-box {
            background: #d4edda;
            border-left: 4px solid #28a745;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .scrollbar::-webkit-scrollbar {
            width: 8px;
        }
        
        .scrollbar::-webkit-scrollbar-track {
            background: #f1f1f1;
            border-radius: 4px;
        }
        
        .scrollbar::-webkit-scrollbar-thumb {
            background: #667eea;
            border-radius: 4px;
        }
        
        .scrollbar::-webkit-scrollbar-thumb:hover {
            background: #555;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 CandleZip HITL</h1>
            <p>Human Text Compression: Benchmark your own Entropy Reduction per Cost!</p>
            <div class="timer" id="timer">⏱ 00:00</div>
        </div>
        
        <div class="main-content">
            <!-- Task Display -->
            <div class="panel task-section">
                <h2>📋 Compression Task</h2>
                <div class="info-box">
                    <strong>Objective:</strong> Predict the next 100-200 words that follow the prefix text below.
                    Use MCP tools to search for the exact source or similar content to extract verbatim continuations.
                </div>
                
                <h3 style="margin-top: 20px; margin-bottom: 10px;">Current Prefix:</h3>
                <div class="text-display scrollbar" id="prefixDisplay"></div>
                
                <div id="memorySection" style="display: none; margin-top: 20px;">
                    <h3 style="margin-bottom: 10px;">Prior Memory (Earlier Chunks):</h3>
                    <div class="text-display scrollbar" id="memoryDisplay"></div>
                </div>
            </div>
            
            <!-- MCP Tools -->
            <div class="panel">
                <h2>🛠 MCP Tools</h2>
                <div class="warning-box">
                    <strong>Tip:</strong> Click a tool to see its parameters and invoke it. Results appear in the history panel.
                </div>
                <div class="tool-list scrollbar" id="toolList"></div>
            </div>
            
            <!-- Tool Invocation -->
            <div class="panel">
                <h2>⚙️ Tool Invocation</h2>
                <div id="toolFormContainer">
                    <p style="color: #666; text-align: center; padding: 40px;">
                        Select a tool from the left panel to invoke it.
                    </p>
                </div>
            </div>
            
            <!-- Tool History -->
            <div class="panel">
                <h2>📜 Tool History</h2>
                <div class="tool-history scrollbar" id="toolHistory">
                    <p style="color: #666; text-align: center; padding: 40px;">
                        No tools invoked yet. Tool results will appear here.
                    </p>
                </div>
            </div>
            
            <!-- Final Prediction -->
            <div class="panel prediction-section">
                <h2>✍️ Your Prediction (Plain Text Only)</h2>
                <div class="warning-box">
                    <strong>Important:</strong> Enter only the plain text continuation. 
                    No markdown, no analysis, no headings. Just the immediate words that would naturally follow the prefix.
                </div>
                <textarea id="finalPrediction" placeholder="Enter your prediction for what comes next..."></textarea>
                <br>
                <button class="btn submit-btn" onclick="submitPrediction()">
                    🚀 Submit Prediction
                </button>
            </div>
        </div>
    </div>
    
    <script>
        let startTime = Date.now();
        let tools = [];
        let selectedTool = null;
        
        // Update timer
        setInterval(() => {
            const elapsed = Math.floor((Date.now() - startTime) / 1000);
            const minutes = Math.floor(elapsed / 60);
            const seconds = elapsed % 60;
            document.getElementById('timer').textContent = 
                `⏱ ${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        }, 1000);
        
        // Load initial data
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                // Display prefix
                document.getElementById('prefixDisplay').textContent = data.prefix;
                
                // Display memory if available
                if (data.prior_memory && data.prior_memory.trim()) {
                    document.getElementById('memorySection').style.display = 'block';
                    document.getElementById('memoryDisplay').textContent = data.prior_memory;
                }
                
                // Load tools
                tools = data.tools;
                renderTools();
                
            } catch (error) {
                console.error('Error loading data:', error);
                alert('Error loading data. Check console for details.');
            }
        }
        
        // Render tool list
        function renderTools() {
            const toolList = document.getElementById('toolList');
            
            if (tools.length === 0) {
                toolList.innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No MCP tools available.</p>';
                return;
            }
            
            toolList.innerHTML = tools.map((tool, index) => `
                <div class="tool-item" onclick="selectTool(${index})">
                    <div class="tool-name">${tool.name}</div>
                    <div class="tool-description">${tool.description || 'No description'}</div>
                </div>
            `).join('');
        }
        
        // Select a tool
        function selectTool(index) {
            selectedTool = tools[index];
            
            // Update active state
            document.querySelectorAll('.tool-item').forEach((item, i) => {
                item.classList.toggle('active', i === index);
            });
            
            // Render tool form
            renderToolForm(selectedTool);
        }
        
        // Render tool invocation form
        function renderToolForm(tool) {
            const container = document.getElementById('toolFormContainer');
            const schema = tool.args_schema || {};
            const properties = schema.properties || {};
            
            let formHtml = `
                <div class="success-box">
                    <strong>${tool.name}</strong><br>
                    ${tool.description || 'No description'}
                </div>
            `;
            
            if (Object.keys(properties).length === 0) {
                formHtml += '<p style="color: #666;">This tool requires no parameters.</p>';
            } else {
                formHtml += Object.entries(properties).map(([name, prop]) => `
                    <div class="form-group">
                        <label for="arg_${name}">${name}</label>
                        ${prop.type === 'string' && (prop.description || '').toLowerCase().includes('url') ? 
                            `<input type="url" id="arg_${name}" placeholder="${prop.description || name}">` :
                            prop.type === 'number' || prop.type === 'integer' ?
                            `<input type="number" id="arg_${name}" placeholder="${prop.description || name}">` :
                            (prop.description || '').length > 50 ?
                            `<textarea id="arg_${name}" placeholder="${prop.description || name}"></textarea>` :
                            `<input type="text" id="arg_${name}" placeholder="${prop.description || name}">`
                        }
                    </div>
                `).join('');
            }
            
            formHtml += `
                <button class="btn" onclick="invokeTool()" style="margin-right: 10px;">
                    ▶️ Invoke Tool
                </button>
                <button class="btn btn-secondary" onclick="clearToolSelection()">
                    ✖ Cancel
                </button>
            `;
            
            container.innerHTML = formHtml;
        }
        
        // Clear tool selection
        function clearToolSelection() {
            selectedTool = null;
            document.querySelectorAll('.tool-item').forEach(item => {
                item.classList.remove('active');
            });
            document.getElementById('toolFormContainer').innerHTML = `
                <p style="color: #666; text-align: center; padding: 40px;">
                    Select a tool from the left panel to invoke it.
                </p>
            `;
        }
        
        // Invoke tool
        async function invokeTool() {
            if (!selectedTool) {
                alert('No tool selected');
                return;
            }
            
            // Collect arguments
            const schema = selectedTool.args_schema || {};
            const properties = schema.properties || {};
            const args = {};
            
            for (const [name, prop] of Object.entries(properties)) {
                const input = document.getElementById(`arg_${name}`);
                if (input) {
                    let value = input.value;
                    if (prop.type === 'number' || prop.type === 'integer') {
                        value = parseFloat(value) || 0;
                    }
                    args[name] = value;
                }
            }
            
            // Show loading state
            const btn = event.target;
            const originalText = btn.textContent;
            btn.disabled = true;
            btn.innerHTML = '<span class="loading"></span> Invoking...';
            
            try {
                const response = await fetch('/api/invoke', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        tool_name: selectedTool.name,
                        args: args
                    })
                });
                
                const result = await response.json();
                
                // Add to history
                addToHistory(selectedTool.name, args, result);
                
                // Show result alert
                if (result.success) {
                    alert('Tool invoked successfully! Check the history panel for results.');
                } else {
                    alert(`Tool invocation failed: ${result.error}`);
                }
                
            } catch (error) {
                console.error('Error invoking tool:', error);
                alert('Error invoking tool. Check console for details.');
            } finally {
                btn.disabled = false;
                btn.textContent = originalText;
            }
        }
        
        // Add to tool history
        function addToHistory(toolName, args, result) {
            const historyContainer = document.getElementById('toolHistory');
            
            // Remove placeholder if exists
            if (historyContainer.textContent.includes('No tools invoked')) {
                historyContainer.innerHTML = '';
            }
            
            const timestamp = new Date().toLocaleTimeString();
            const historyItem = document.createElement('div');
            historyItem.className = 'history-item';
            
            let argsStr = Object.keys(args).length > 0 ? 
                JSON.stringify(args, null, 2) : 'No arguments';
            
            let resultStr = result.success ? 
                result.result : 
                `Error: ${result.error}`;
            
            historyItem.innerHTML = `
                <div class="history-header">
                    🔧 ${toolName} <span style="float: right; color: #666; font-weight: normal;">${timestamp}</span>
                </div>
                <div style="margin: 10px 0;">
                    <strong>Arguments:</strong>
                    <div class="history-result">${argsStr}</div>
                </div>
                <div>
                    <strong>Result:</strong>
                    <div class="history-result">${resultStr}</div>
                </div>
            `;
            
            historyContainer.insertBefore(historyItem, historyContainer.firstChild);
        }
        
        // Submit prediction
        async function submitPrediction() {
            const prediction = document.getElementById('finalPrediction').value.trim();
            
            if (!prediction) {
                alert('Please enter a prediction before submitting.');
                return;
            }
            
            if (!confirm('Are you sure you want to submit this prediction? This will close the interface.')) {
                return;
            }
            
            try {
                const response = await fetch('/api/submit', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        final_text: prediction
                    })
                });
                
                const result = await response.json();
                
                if (result.success) {
                    document.body.innerHTML = `
                        <div style="display: flex; align-items: center; justify-content: center; height: 100vh; flex-direction: column;">
                            <h1 style="color: white; font-size: 3em;">✅ Prediction Submitted!</h1>
                            <p style="color: white; font-size: 1.5em; margin-top: 20px;">You can close this window now.</p>
                        </div>
                    `;
                    
                    // Close window after delay
                    setTimeout(() => {
                        window.close();
                    }, 2000);
                } else {
                    alert('Error submitting prediction: ' + result.error);
                }
                
            } catch (error) {
                console.error('Error submitting prediction:', error);
                alert('Error submitting prediction. Check console for details.');
            }
        }
        
        // Initialize
        loadData();
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Render the main interface."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/data')
def get_data():
    """Get task data and available tools."""
    # Parse the task to extract prefix and memory
    task_text = task_data["task"]
    
    # Extract prefix (between "Current document prefix" and "Output:")
    prefix_start = task_text.find("Current document prefix (UTF-8 text):")
    output_start = task_text.find("\n\nOutput:")
    
    if prefix_start != -1 and output_start != -1:
        prefix = task_text[prefix_start + len("Current document prefix (UTF-8 text):"):output_start].strip()
    else:
        prefix = task_text
    
    # Extract prior memory (between "Prior memory" and "Current document prefix")
    memory_start = task_text.find("Prior memory (from earlier chunks):")
    memory_end = prefix_start if prefix_start != -1 else -1
    
    prior_memory = ""
    if memory_start != -1 and memory_end != -1:
        prior_memory = task_text[memory_start + len("Prior memory (from earlier chunks):"):memory_end].strip()
    
    task_data["prefix"] = prefix
    task_data["prior_memory"] = prior_memory
    
    return jsonify({
        "task": task_data["task"],
        "prefix": prefix,
        "prior_memory": prior_memory,
        "tools": tool_manager.get_tool_list() if tool_manager else []
    })


@app.route('/api/invoke', methods=['POST'])
def invoke_tool():
    """Invoke an MCP tool."""
    data = request.json
    tool_name = data.get("tool_name")
    args = data.get("args", {})
    
    if not tool_manager:
        return jsonify({"success": False, "error": "Tool manager not initialized"})
    
    result = tool_manager.invoke_tool(tool_name, args)
    
    # Add to history
    task_data["tool_history"].append({
        "tool": tool_name,
        "args": args,
        "result": result,
        "timestamp": time.time()
    })
    
    return jsonify(result)


@app.route('/api/submit', methods=['POST'])
def submit_prediction():
    """Submit final prediction."""
    data = request.json
    final_text = data.get("final_text", "")
    
    # Record final text
    task_data["final_text"] = final_text
    task_data["completed"] = True
    
    # Emit AGENT_RESULT_JSON immediately so CandleZip can proceed without waiting
    global printed_result
    try:
        duration_ms = int((time.time() - task_data["start_time"]) * 1000)
        result_json = {"final_text": final_text, "duration_ms": duration_ms}
        print(f"AGENT_RESULT_JSON:{json.dumps(result_json)}")
        sys.stdout.flush()
        printed_result = True
    except Exception as e:
        print(f"[HITL] Failed to emit AGENT_RESULT_JSON: {e}", file=sys.stderr)
    
    # Trigger graceful shutdown by calling internal endpoint within a request context
    def _delayed_shutdown():
        try:
            import urllib.request
            url = f"http://{app.config.get('SERVER_HOST','localhost')}:{app.config.get('SERVER_PORT',5000)}/_shutdown"
            req = urllib.request.Request(url, method='POST')
            with urllib.request.urlopen(req, data=b'') as _:
                pass
        except Exception as e:
            print(f"[HITL] Shutdown request failed: {e}", file=sys.stderr)
    threading.Timer(0.4, _delayed_shutdown).start()
    
    # Hard exit fallback in case werkzeug shutdown is unavailable
    def _hard_exit():
        try:
            import os as _os, signal as _signal
            _os._exit(0)
        except Exception:
            pass
    threading.Timer(2.0, _hard_exit).start()
    
    return jsonify({"success": True})


@app.route('/_shutdown', methods=['POST'])
def shutdown_server():
    """Shutdown the Flask development server (must be called via HTTP)."""
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()
        return 'OK'
    # Fallback: terminate process (unlikely needed for Werkzeug)
    import os as _os, signal as _signal
    _os.kill(_os.getpid(), _signal.SIGINT)
    return 'OK'


# ============================================================================
# Main Entry Point
# ============================================================================

def main() -> int:
    """Main entry point - matches agent_v2.py interface."""
    # Load environment variables
    env_path = find_dotenv()
    load_dotenv(env_path if env_path else None)
    
    # Parse arguments (same as agent_v2.py)
    parser = argparse.ArgumentParser(
        description="Human-in-the-Loop Agent for CandleZip"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--task",
        help="Task description for the human to complete"
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
        help="Maximum steps (not used for HITL, kept for compatibility)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port for the web interface"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open browser"
    )
    
    args = parser.parse_args()
    
    # Resolve task text
    task_text: str
    if getattr(args, "task_file", None):
        with open(args.task_file, "r", encoding="utf-8") as f:
            task_text = f.read()
    else:
        task_text = args.task
    
    # Initialize global state
    task_data["task"] = task_text
    task_data["start_time"] = time.time()
    
    # Initialize MCP tools
    global tool_manager
    tool_manager = MCPToolManager(args.mcp_config)
    tool_manager.load_tools()
    
    # Resolve a free port (fallback if requested port is busy)
    desired_port = int(args.port)
    actual_port = desired_port
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(('localhost', desired_port))
            except OSError:
                # find ephemeral free port
                s.bind(('localhost', 0))
                actual_port = s.getsockname()[1]
    except Exception:
        actual_port = desired_port

    # Start web server in background
    print(f"[HITL] Starting web interface on http://localhost:{actual_port}", file=sys.stderr)
    print(f"[HITL] Opening browser...", file=sys.stderr)

    # Open browser
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{actual_port}")).start()
    
    try:
        # Share host/port with shutdown helper
        app.config['SERVER_HOST'] = 'localhost'
        app.config['SERVER_PORT'] = int(actual_port)
        # Run Flask app (blocks until shutdown)
        app.run(host='localhost', port=actual_port, debug=False, use_reloader=False)
        
        # After shutdown, emit only if not already printed inside submit handler
        if printed_result:
            tool_manager.cleanup()
            return 0
        if task_data["completed"] and task_data["final_text"]:
            duration_ms = int((time.time() - task_data["start_time"]) * 1000)
            final_text = task_data["final_text"]
            result_json = {"final_text": final_text, "duration_ms": duration_ms}
            print(f"AGENT_RESULT_JSON:{json.dumps(result_json)}")
            tool_manager.cleanup()
            return 0
        print("Error: No prediction submitted", file=sys.stderr)
        tool_manager.cleanup()
        return 1
            
    except KeyboardInterrupt:
        print("\n[HITL] Interrupted by user", file=sys.stderr)
        tool_manager.cleanup()
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if tool_manager:
            tool_manager.cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())

