#!/usr/bin/env python3
"""
CandleZip Science Tool: Comprehensive Benchmarking Framework

This tool provides automated benchmarking across domains, toolsets, and agents
for the Cashin Complexity research framework. It handles data management,
experiment execution, and result aggregation for direct paper integration.

Usage:
    python science_tool.py --help
    python science_tool.py run --config benchmark_config.json
    python science_tool.py add-domain --name academic --file paper.txt
    python science_tool.py list-domains
    python science_tool.py generate-tables --output results/tables.tex

Design Principles:
    - Single Responsibility: Each class handles one aspect of benchmarking
    - Open/Closed: Extensible without modifying core logic
    - Liskov Substitution: Abstract interfaces for domains/toolsets
    - Interface Segregation: Clean separation of concerns
    - Dependency Inversion: Depends on abstractions, not concretions
"""

import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import csv
import shutil
from abc import ABC, abstractmethod
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import gzip
import tempfile


# ============================================================================
# Domain Abstractions (Interface Segregation Principle)
# ============================================================================

@dataclass
class BenchmarkFile:
    """Single file in a domain for benchmarking."""
    path: Path
    name: str
    size_bytes: int
    
    @classmethod
    def from_path(cls, path: Path) -> 'BenchmarkFile':
        """Create from file path with automatic size detection."""
        if not path.exists():
            raise FileNotFoundError(f"Benchmark file not found: {path}")
        return cls(
            path=path.resolve(),
            name=path.name,
            size_bytes=path.stat().st_size
        )


@dataclass
class Domain:
    """A domain represents a category of test data (e.g., 'academic', 'wikipedia')."""
    name: str
    files: List[BenchmarkFile] = field(default_factory=list)
    
    def add_file(self, file_path: Path) -> None:
        """Add a benchmark file to this domain."""
        self.files.append(BenchmarkFile.from_path(file_path))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'files': [
                {'path': str(f.path), 'name': f.name, 'size_bytes': f.size_bytes}
                for f in self.files
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Domain':
        """Deserialize from dictionary."""
        domain = cls(name=data['name'])
        for f in data['files']:
            domain.files.append(BenchmarkFile(
                path=Path(f['path']),
                name=f['name'],
                size_bytes=f['size_bytes']
            ))
        return domain


# ============================================================================
# SIMDL v1.1: Offline Lambda-Sweeps and Analysis Functions
# ============================================================================

def sweep_lambda(rows: List[Dict[str, Any]], price_col: str, lambdas: List[float]) -> List[Dict[str, Any]]:
    """
    Perform offline lambda-sweep without re-encoding.
    
    Args:
        rows: List of CSV row dictionaries from proof.csv
        price_col: Price column to use ('price_transcript_bits', 'price_pointer_bits', 'agent_duration_ms')
        lambdas: List of lambda values to sweep
        
    Returns:
        List of sweep points with keys: lambda, bits, cost
    """
    curve = []
    for lam in lambdas:
        total_bits, total_cost = 0.0, 0.0
        for r in rows:
            b = float(r['cross_entropy_baseline_bits'])
            bprime = float(r['cross_entropy_conditioned_bits'])
            g = float(r['gate_bits'])
            c = float(r[price_col]) if price_col != 'agent_duration_ms' else float(r['agent_duration_ms'])
            
            tool_bits = bprime + g + lam * c
            if tool_bits < b:
                total_bits += tool_bits
                total_cost += c
            else:
                total_bits += b
                
        curve.append({'lambda': lam, 'bits': total_bits, 'cost': total_cost})
    return curve

def compute_pareto_frontier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Compute Pareto frontier (lower envelope) from (cost, bits) points.
    
    Args:
        points: List of (cost, bits) tuples
        
    Returns:
        List of Pareto-optimal (cost, bits) tuples sorted by cost
    """
    if len(points) < 2:
        return points
        
    # Sort by cost
    sorted_points = sorted(points, key=lambda x: x[0])
    
    # Build lower envelope
    frontier = []
    for cost, bits in sorted_points:
        # Remove dominated points (higher cost and higher bits)
        while frontier and frontier[-1][1] >= bits:
            frontier.pop()
        frontier.append((cost, bits))
    
    return frontier

def bootstrap_ci(rows: List[Dict[str, Any]], stat_fn, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    """
    Compute bootstrap confidence intervals.
    
    Args:
        rows: List of CSV row dictionaries
        stat_fn: Function to compute statistic from sample
        n_boot: Number of bootstrap samples
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (lower_95ci, upper_95ci)
    """
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = [rng.choice(rows) for _ in rows]
        vals.append(stat_fn(sample))
    vals.sort()
    return vals[int(0.025*len(vals))], vals[int(0.975*len(vals))]

def run_baseline_compressor(input_file: Path, compressor: str) -> Optional[int]:
    """
    Run baseline compressor (gzip or zstd) and return compressed size in bytes.
    
    Args:
        input_file: Path to input file
        compressor: 'gzip' or 'zstd'
        
    Returns:
        Compressed size in bytes, or None if failed
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
        
        if compressor == 'gzip':
            # Use gzip command line tool
            # Prefer gzip.exe on Windows if present
            exe = 'gzip.exe' if os.name == 'nt' else 'gzip'
            result = subprocess.run([exe, '-9', '-c', str(input_file)], 
                                   stdout=open(tmp_path, 'wb'), stderr=subprocess.PIPE)
        elif compressor == 'zstd':
            # Use zstd command line tool
            exe = 'zstd.exe' if os.name == 'nt' else 'zstd'
            result = subprocess.run([exe, '-19', '--long=31', '-o', str(tmp_path), str(input_file)], 
                                   stderr=subprocess.PIPE)
        else:
            return None
            
        if result.returncode == 0:
            size = tmp_path.stat().st_size
            tmp_path.unlink()  # Clean up
            return size
        else:
            tmp_path.unlink()  # Clean up
            return None
            
    except Exception as e:
        print(f"Warning: Failed to run {compressor}: {e}")
        return None

# ============================================================================
# Configuration Management (Single Responsibility)
# ============================================================================

@dataclass
class ToolsetConfig:
    """Configuration for a specific MCP toolset."""
    name: str
    mcp_config_path: Path
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'mcp_config_path': str(self.mcp_config_path),
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolsetConfig':
        return cls(
            name=data['name'],
            mcp_config_path=Path(data['mcp_config_path']),
            description=data.get('description', '')
        )


@dataclass
class AgentConfig:
    """Configuration for a specific agent."""
    name: str
    script_path: Path
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'script_path': str(self.script_path),
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        return cls(
            name=data['name'],
            script_path=Path(data['script_path']),
            description=data.get('description', '')
        )


@dataclass
class BenchmarkConfig:
    """Complete benchmark configuration."""
    domains: List[Domain] = field(default_factory=list)
    toolsets: List[ToolsetConfig] = field(default_factory=list)
    agents: List[AgentConfig] = field(default_factory=list)
    
    # CandleZip parameters
    backend: str = "smollm"
    context: int = 512
    reprime_interval: int = 512
    scan_lookahead: int = 512
    scan_max_steps: int = 12
    scan_agent_timeout: int = 300
    
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'domains': [d.to_dict() for d in self.domains],
            'toolsets': [t.to_dict() for t in self.toolsets],
            'agents': [a.to_dict() for a in self.agents],
            'backend': self.backend,
            'context': self.context,
            'reprime_interval': self.reprime_interval,
            'scan_lookahead': self.scan_lookahead,
            'scan_max_steps': self.scan_max_steps,
            'scan_agent_timeout': self.scan_agent_timeout,
            'output_dir': str(self.output_dir)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BenchmarkConfig':
        return cls(
            domains=[Domain.from_dict(d) for d in data.get('domains', [])],
            toolsets=[ToolsetConfig.from_dict(t) for t in data.get('toolsets', [])],
            agents=[AgentConfig.from_dict(a) for a in data.get('agents', [])],
            backend=data.get('backend', 'smollm'),
            context=data.get('context', 512),
            reprime_interval=data.get('reprime_interval', 512),
            scan_lookahead=data.get('scan_lookahead', 512),
            scan_max_steps=data.get('scan_max_steps', 12),
            scan_agent_timeout=data.get('scan_agent_timeout', 300),
            output_dir=Path(data.get('output_dir', 'benchmark_results'))
        )
    
    def save(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> 'BenchmarkConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# ============================================================================
# Experiment Execution (Single Responsibility)
# ============================================================================

@dataclass
class ExperimentResult:
    """Result from a single benchmark run."""
    domain: str
    file: str
    toolset: str
    agent: str
    timestamp: str
    
    # Results
    success: bool
    error_message: str = ""
    
    # Metrics (from proof.csv)
    total_chunks: int = 0
    gated_chunks: int = 0
    total_bits_saved: float = 0.0
    total_baseline_bits: float = 0.0
    percent_saved_overall: float = 0.0
    total_duration_ms: int = 0
    total_agent_calls: int = 0
    
    # File metrics
    original_size: int = 0
    compressed_size: int = 0
    bits_per_byte: float = 0.0
    
    # Roundtrip validation
    roundtrip_success: bool = False
    roundtrip_identical: bool = False
    
    # Paths
    output_dir: Optional[Path] = None
    proof_csv_path: Optional[Path] = None
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        if self.output_dir:
            d['output_dir'] = str(self.output_dir)
        if self.proof_csv_path:
            d['proof_csv_path'] = str(self.proof_csv_path)
        return d


class CandleZipRunner:
    """Handles execution of CandleZip binary with proper error handling."""
    
    def __init__(self, binary_path: Optional[Path] = None):
        """Initialize runner with binary path detection."""
        self.binary_path = self._find_binary(binary_path)
    
    def _find_binary(self, explicit_path: Optional[Path]) -> Path:
        """Find CandleZip binary, preferring explicit path."""
        if explicit_path and explicit_path.exists():
            return explicit_path
        
        # Common locations
        candidates = [
            Path("target/release/candlezip.exe"),
            Path("target/release/candlezip"),
            Path("candlezip.exe"),
            Path("candlezip"),
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
        
        raise FileNotFoundError(
            "CandleZip binary not found. Build it first with: cargo build --release"
        )
    
    def run_self_test(
        self,
        input_file: Path,
        config: BenchmarkConfig,
        agent: AgentConfig,
        toolset: ToolsetConfig,
        output_dir: Path
    ) -> Tuple[bool, str]:
        """
        Run CandleZip self-test (encode + decode with --reuse).
        
        Returns:
            (success: bool, error_message: str)
        """
        # Prepare environment
        env = os.environ.copy()
        
        # Build command
        cmd = [
            str(self.binary_path),
            "self-test",
            "--backend", config.backend,
            "--agent",
            "--scan",
            "--scan-lookahead", str(config.scan_lookahead),
            "--context", str(config.context),
            "--reprime-interval", str(config.reprime_interval),
            "--scan-agent-script", str(agent.script_path),
            "--scan_max_steps", str(config.scan_max_steps),
            "--scan-agent-timeout", str(config.scan_agent_timeout),
            "--scan-mcp-config", str(toolset.mcp_config_path),
            "--scan_output_dir", str(output_dir),
            str(input_file)
        ]
        
        try:
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour max
            )
            
            if result.returncode == 0:
                return True, ""
            else:
                error = f"Exit code {result.returncode}\nStdout: {result.stdout}\nStderr: {result.stderr}"
                return False, error
                
        except subprocess.TimeoutExpired:
            return False, "Timeout (1 hour exceeded)"
        except Exception as e:
            return False, f"Exception: {str(e)}"


class ResultsAggregator:
    """Aggregates experiment results and generates reports."""
    
    @staticmethod
    def parse_proof_csv(csv_path: Path) -> Dict[str, Any]:
        """Parse proof.csv and extract summary statistics with SIMDL v1.1 support."""
        if not csv_path.exists():
            return {}
        
        total_chunks = 0
        gated_chunks = 0
        total_bits_saved = 0.0
        total_baseline_bits = 0.0
        total_duration_ms = 0
        total_agent_calls = 0
        rows = []
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
                total_chunks += 1
                if int(row.get('gate', 0)) == 1:
                    gated_chunks += 1
                
                # SIMDL v1.1: Include ALL bits_saved, not just positive
                bits_saved = float(row.get('bits_saved', 0))
                total_bits_saved += bits_saved  # No filtering for negative values
                
                total_baseline_bits += float(row.get('cross_entropy_baseline_bits', 0))
                total_duration_ms += int(row.get('agent_duration_ms', 0))
                total_agent_calls += int(row.get('agent_calls', 0))
        
        percent_saved = (total_bits_saved / total_baseline_bits * 100.0 
                        if total_baseline_bits > 0 else 0.0)
        
        return {
            'total_chunks': total_chunks,
            'gated_chunks': gated_chunks,
            'total_bits_saved': total_bits_saved,
            'total_baseline_bits': total_baseline_bits,
            'percent_saved_overall': percent_saved,
            'total_duration_ms': total_duration_ms,
            'total_agent_calls': total_agent_calls,
            'rows': rows  # SIMDL v1.1: Return raw rows for offline analysis
        }
    
    @staticmethod
    def load_all_proof_csvs(csv_paths: List[Path]) -> List[Dict[str, Any]]:
        """Load and combine all proof.csv files into a single list of rows."""
        all_rows = []
        for csv_path in csv_paths:
            if csv_path.exists():
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        all_rows.append(row)
        return all_rows
    
    @staticmethod
    def generate_pareto_curves(csv_paths: List[Path], group_by: List[str], price_col: str, 
                              lambdas: List[float], output_dir: Path) -> None:
        """Generate Pareto curves for different groups and save to CSV."""
        all_rows = ResultsAggregator.load_all_proof_csvs(csv_paths)
        
        # Group rows
        groups = {}
        for row in all_rows:
            group_key = tuple(row.get(col, 'unknown') for col in group_by)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(row)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for group_key, group_rows in groups.items():
            # Perform lambda sweep
            sweep_results = sweep_lambda(group_rows, price_col, lambdas)
            
            # Extract (cost, bits) points
            points = [(r['cost'], r['bits']) for r in sweep_results]
            frontier = compute_pareto_frontier(points)
            
            # Save sweep results
            group_name = '_'.join(str(k) for k in group_key)
            sweep_file = output_dir / f"sweep_{group_name}_{price_col}.csv"
            with open(sweep_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['lambda', 'bits', 'cost'])
                writer.writeheader()
                writer.writerows(sweep_results)
            
            # Save frontier
            frontier_file = output_dir / f"frontier_{group_name}_{price_col}.csv"
            with open(frontier_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['cost', 'bits'])
                writer.writeheader()
                for cost, bits in frontier:
                    writer.writerow({'cost': cost, 'bits': bits})
    
    @staticmethod
    def compute_confidence_intervals(csv_path: Path, price_col: str, lambda_val: float, 
                                   metric: str, n_boot: int = 2000) -> Dict[str, Any]:
        """Compute bootstrap confidence intervals for a specific metric."""
        all_rows = ResultsAggregator.load_all_proof_csvs([csv_path])
        
        if metric == 'net_bits':
            def stat_fn(sample):
                result = sweep_lambda(sample, price_col, [lambda_val])
                return result[0]['bits'] if result else 0
        elif metric == 'gate_rate':
            def stat_fn(sample):
                gated = sum(1 for r in sample if float(r.get('gate_bits', 0)) > 0)
                return gated / len(sample) if sample else 0
        elif metric == 'mean_advantage':
            def stat_fn(sample):
                advantages = []
                for r in sample:
                    b = float(r['cross_entropy_baseline_bits'])
                    bprime = float(r['cross_entropy_conditioned_bits'])
                    g = float(r['gate_bits'])
                    advantages.append(b - (bprime + g))
                return sum(advantages) / len(advantages) if advantages else 0
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        ci_low, ci_high = bootstrap_ci(all_rows, stat_fn, n_boot)
        point_estimate = stat_fn(all_rows)
        
        return {
            'metric': metric,
            'price_col': price_col,
            'lambda': lambda_val,
            'point_estimate': point_estimate,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_boot': n_boot
        }
    
    @staticmethod
    def generate_pareto_plot(frontier_files: List[Path], output_path: Path) -> None:
        """Generate Pareto frontier plot from CSV files."""
        plt.figure(figsize=(10, 6))
        
        for frontier_file in frontier_files:
            if frontier_file.exists():
                with open(frontier_file, 'r') as f:
                    reader = csv.DictReader(f)
                    costs, bits = [], []
                    for row in reader:
                        costs.append(float(row['cost']))
                        bits.append(float(row['bits']))
                
                # Extract method name from filename
                method_name = frontier_file.stem.replace('frontier_', '').replace('_', ' ')
                plt.plot(costs, bits, 'o-', label=method_name, linewidth=2, markersize=4)
        
        plt.xlabel('Cost')
        plt.ylabel('Bits')
        plt.title('Pareto Frontiers: Bits vs Cost')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def generate_tool_roi_plot(csv_paths: List[Path], price_col: str, output_path: Path) -> None:
        """Generate Tool ROI (Return on Investment) bar plot."""
        all_rows = ResultsAggregator.load_all_proof_csvs(csv_paths)
        
        # Group by tool_id_best
        tool_rois = {}
        for row in all_rows:
            tool_id = row.get('tool_id_best', 'none')
            if tool_id == 'none':
                continue
                
            b = float(row['cross_entropy_baseline_bits'])
            bprime = float(row['cross_entropy_conditioned_bits'])
            g = float(row['gate_bits'])
            price = float(row[price_col]) if price_col != 'agent_duration_ms' else float(row['agent_duration_ms'])
            
            if price > 0:
                roi = (b - (bprime + g)) / price
                if tool_id not in tool_rois:
                    tool_rois[tool_id] = []
                tool_rois[tool_id].append(roi)
        
        # Compute mean ROI and CI for each tool
        tool_names = []
        mean_rois = []
        ci_lows = []
        ci_highs = []
        
        for tool_id, rois in tool_rois.items():
            if len(rois) >= 2:  # Need at least 2 samples for CI
                tool_names.append(tool_id)
                mean_roi = np.mean(rois)
                mean_rois.append(mean_roi)
                
                # Compute CI using bootstrap
                def roi_stat_fn(sample):
                    return np.mean([r for r in sample])
                
                ci_low, ci_high = bootstrap_ci([{'roi': r} for r in rois], 
                                             lambda s: np.mean([x['roi'] for x in s]))
                ci_lows.append(ci_low - mean_roi)
                ci_highs.append(ci_high - mean_roi)
        
        if tool_names:
            plt.figure(figsize=(12, 6))
            x_pos = np.arange(len(tool_names))
            bars = plt.bar(x_pos, mean_rois, yerr=[ci_lows, ci_highs], 
                          capsize=5, alpha=0.7, color='skyblue', edgecolor='navy')
            
            plt.xlabel('Tool ID')
            plt.ylabel('ROI (bits saved per cost unit)')
            plt.title(f'Tool ROI with 95% Confidence Intervals ({price_col})')
            plt.xticks(x_pos, tool_names, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            plt.axhline(y=0, color='red', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
    
    @staticmethod
    def generate_advantage_histogram(csv_paths: List[Path], lambda_val: float, price_col: str, output_path: Path) -> None:
        """Generate advantage histogram for a specific lambda value."""
        all_rows = ResultsAggregator.load_all_proof_csvs(csv_paths)
        
        advantages = []
        for row in all_rows:
            b = float(row['cross_entropy_baseline_bits'])
            bprime = float(row['cross_entropy_conditioned_bits'])
            g = float(row['gate_bits'])
            c = float(row[price_col]) if price_col != 'agent_duration_ms' else float(row['agent_duration_ms'])
            
            advantage = (b - (bprime + g)) - lambda_val * c
            advantages.append(advantage)
        
        plt.figure(figsize=(10, 6))
        plt.hist(advantages, bins=30, alpha=0.7, color='lightgreen', edgecolor='darkgreen')
        plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Break-even')
        plt.xlabel(f'Advantage (bits) at λ={lambda_val}')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Advantage at λ={lambda_val} ({price_col})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def generate_latex_table(results: List[ExperimentResult], output_path: Path) -> None:
        """Generate LaTeX table from results."""
        with open(output_path, 'w') as f:
            f.write("\\begin{table}[H]\n")
            f.write("\\centering\n")
            f.write("\\begin{tabular}{llrrrr}\n")
            f.write("\\toprule\n")
            f.write("Domain & Toolset & Agent & Bits Saved & \\% Improvement & Gate Rate \\\\\n")
            f.write("\\midrule\n")
            
            for result in results:
                if result.success:
                    gate_rate = (f"{result.gated_chunks}/{result.total_chunks}" 
                                if result.total_chunks > 0 else "N/A")
                    f.write(f"{result.domain} & {result.toolset} & {result.agent} & "
                           f"{result.total_bits_saved:.2f} & {result.percent_saved_overall:.1f}\\% & "
                           f"{gate_rate} \\\\\n")
            
            f.write("\\bottomrule\n")
            f.write("\\end{tabular}\n")
            f.write("\\caption{Benchmark results across domains, toolsets, and agents.}\n")
            f.write("\\label{tab:benchmark-results}\n")
            f.write("\\end{table}\n")
    
    @staticmethod
    def save_results_json(results: List[ExperimentResult], output_path: Path) -> None:
        """Save complete results to JSON for further analysis."""
        with open(output_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
    
    @staticmethod
    def print_summary(results: List[ExperimentResult]) -> None:
        """Print human-readable summary to console."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        total = len(results)
        successful = sum(1 for r in results if r.success)
        
        print(f"\nTotal experiments: {total}")
        print(f"Successful: {successful}")
        print(f"Failed: {total - successful}")
        
        if successful > 0:
            print("\n" + "-"*80)
            print(f"{'Domain':<15} {'Toolset':<15} {'Agent':<15} {'Bits Saved':<12} {'% Improv':<10} {'Gates':<8}")
            print("-"*80)
            
            for result in results:
                if result.success:
                    gate_rate = f"{result.gated_chunks}/{result.total_chunks}"
                    print(f"{result.domain:<15} {result.toolset:<15} {result.agent:<15} "
                          f"{result.total_bits_saved:>11.2f} {result.percent_saved_overall:>9.1f}% "
                          f"{gate_rate:<8}")
        
        print("="*80 + "\n")


# ============================================================================
# Benchmark Orchestration (Dependency Inversion)
# ============================================================================

class BenchmarkOrchestrator:
    """
    Main orchestrator for running benchmarks across domains/toolsets/agents.
    
    This class coordinates the entire benchmarking process:
    1. Iterates through configured domains
    2. For each domain, runs tests with current toolset
    3. Optionally switches agents if configured
    4. Aggregates and reports results
    """
    
    def __init__(self, config: BenchmarkConfig, runner: CandleZipRunner):
        self.config = config
        self.runner = runner
        self.results: List[ExperimentResult] = []
    
    def run_all(self) -> List[ExperimentResult]:
        """Run all configured benchmarks."""
        print(f"\n{'='*80}")
        print(f"Starting Benchmark Suite")
        print(f"{'='*80}")
        print(f"Domains: {len(self.config.domains)}")
        print(f"Toolsets: {len(self.config.toolsets)}")
        print(f"Agents: {len(self.config.agents)}")
        
        total_experiments = sum(
            len(domain.files) 
            for domain in self.config.domains
        ) * len(self.config.toolsets) * len(self.config.agents)
        
        print(f"Total experiments: {total_experiments}")
        print(f"{'='*80}\n")
        
        experiment_count = 0
        
        for toolset in self.config.toolsets:
            print(f"\n{'='*80}")
            print(f"TOOLSET: {toolset.name}")
            print(f"MCP Config: {toolset.mcp_config_path}")
            print(f"{'='*80}\n")
            
            for agent in self.config.agents:
                print(f"\n{'-'*80}")
                print(f"AGENT: {agent.name}")
                print(f"Script: {agent.script_path}")
                print(f"{'-'*80}\n")
                
                for domain in self.config.domains:
                    print(f"\n  Domain: {domain.name} ({len(domain.files)} files)")
                    
                    for bench_file in domain.files:
                        experiment_count += 1
                        print(f"    [{experiment_count}/{total_experiments}] {bench_file.name}... ", end="", flush=True)
                        
                        result = self._run_experiment(
                            domain=domain,
                            bench_file=bench_file,
                            toolset=toolset,
                            agent=agent
                        )
                        
                        self.results.append(result)
                        
                        if result.success:
                            print(f"✓ ({result.percent_saved_overall:.1f}% saved, {result.gated_chunks}/{result.total_chunks} gated)")
                        else:
                            print(f"✗ FAILED: {result.error_message[:50]}")
        
        return self.results
    
    def _run_experiment(
        self,
        domain: Domain,
        bench_file: BenchmarkFile,
        toolset: ToolsetConfig,
        agent: AgentConfig
    ) -> ExperimentResult:
        """Run a single experiment and collect results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = (
            self.config.output_dir / 
            f"{domain.name}_{toolset.name}_{agent.name}_{bench_file.name}_{timestamp}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        
        result = ExperimentResult(
            domain=domain.name,
            file=bench_file.name,
            toolset=toolset.name,
            agent=agent.name,
            timestamp=timestamp,
            success=False,
            original_size=bench_file.size_bytes,
            output_dir=output_dir
        )
        
        # Run self-test (encode + decode with --reuse)
        success, error = self.runner.run_self_test(
            input_file=bench_file.path,
            config=self.config,
            agent=agent,
            toolset=toolset,
            output_dir=output_dir
        )
        
        if not success:
            result.error_message = error
            return result
        
        # Parse results from proof.csv
        proof_csv = self._find_proof_csv(output_dir)
        if proof_csv:
            result.proof_csv_path = proof_csv
            metrics = ResultsAggregator.parse_proof_csv(proof_csv)
            
            result.total_chunks = metrics.get('total_chunks', 0)
            result.gated_chunks = metrics.get('gated_chunks', 0)
            result.total_bits_saved = metrics.get('total_bits_saved', 0.0)
            result.total_baseline_bits = metrics.get('total_baseline_bits', 0.0)
            result.percent_saved_overall = metrics.get('percent_saved_overall', 0.0)
            result.total_duration_ms = metrics.get('total_duration_ms', 0)
            result.total_agent_calls = metrics.get('total_agent_calls', 0)
        
        # Check for compressed output and roundtrip file
        roundtrip_file = bench_file.path.with_suffix(".roundtrip.txt")
        if roundtrip_file.exists():
            result.roundtrip_success = True
            # Verify identical roundtrip
            original_bytes = bench_file.path.read_bytes()
            roundtrip_bytes = roundtrip_file.read_bytes()
            result.roundtrip_identical = (original_bytes == roundtrip_bytes)
        
        result.success = True
        return result
    
    def _find_proof_csv(self, output_dir: Path) -> Optional[Path]:
        """Find proof.csv in output directory (may be in subdirectories)."""
        for path in output_dir.rglob("proof.csv"):
            return path
        return None
    
    def save_results(self) -> None:
        """Save all results to output directory."""
        # Create timestamped results directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = self.config.output_dir / f"aggregated_{timestamp}"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_path = results_dir / "results.json"
        ResultsAggregator.save_results_json(self.results, json_path)
        print(f"\n✓ Results saved to: {json_path}")
        
        # Generate LaTeX table
        latex_path = results_dir / "results_table.tex"
        ResultsAggregator.generate_latex_table(self.results, latex_path)
        print(f"✓ LaTeX table saved to: {latex_path}")
        
        # Print summary
        ResultsAggregator.print_summary(self.results)


# ============================================================================
# CLI Interface
# ============================================================================

def cmd_init(args):
    """Initialize a new benchmark configuration."""
    config = BenchmarkConfig()
    
    # Create example configuration
    config.toolsets.append(ToolsetConfig(
        name="baseline",
        mcp_config_path=Path("agent/mcp_config.json"),
        description="Baseline MCP configuration"
    ))
    
    config.agents.append(AgentConfig(
        name="default",
        script_path=Path("agent/agent_v2.py"),
        description="Default CrewAI agent"
    ))
    
    output_path = Path(args.output)
    config.save(output_path)
    print(f"✓ Created configuration template: {output_path}")
    print(f"  Edit this file to add domains, toolsets, and agents.")


def cmd_add_domain(args):
    """Add a domain and file to existing configuration."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"✗ Configuration not found: {config_path}")
        print(f"  Run: python science_tool.py init")
        return
    
    config = BenchmarkConfig.load(config_path)
    
    # Find or create domain
    domain = None
    for d in config.domains:
        if d.name == args.name:
            domain = d
            break
    
    if domain is None:
        domain = Domain(name=args.name)
        config.domains.append(domain)
        print(f"✓ Created new domain: {args.name}")
    
    # Add file
    file_path = Path(args.file)
    domain.add_file(file_path)
    print(f"✓ Added file to domain '{args.name}': {file_path.name} ({file_path.stat().st_size} bytes)")
    
    # Save updated config
    config.save(config_path)
    print(f"✓ Updated configuration: {config_path}")


def cmd_list_domains(args):
    """List all domains and their files."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"✗ Configuration not found: {config_path}")
        return
    
    config = BenchmarkConfig.load(config_path)
    
    if not config.domains:
        print("No domains configured.")
        return
    
    print("\n" + "="*80)
    print("CONFIGURED DOMAINS")
    print("="*80)
    
    for domain in config.domains:
        print(f"\n{domain.name}:")
        if not domain.files:
            print("  (no files)")
        else:
            for f in domain.files:
                print(f"  - {f.name} ({f.size_bytes} bytes) @ {f.path}")
    
    print("\n" + "="*80 + "\n")


def cmd_run(args):
    """Run benchmarks from configuration."""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"✗ Configuration not found: {config_path}")
        return
    
    config = BenchmarkConfig.load(config_path)
    
    # Validate configuration
    if not config.domains:
        print("✗ No domains configured. Add domains with: python science_tool.py add-domain")
        return
    
    if not config.toolsets:
        print("✗ No toolsets configured. Edit config file to add toolsets.")
        return
    
    if not config.agents:
        print("✗ No agents configured. Edit config file to add agents.")
        return
    
    # Initialize runner
    runner = CandleZipRunner(binary_path=Path(args.binary) if args.binary else None)
    
    # Run benchmarks
    orchestrator = BenchmarkOrchestrator(config, runner)
    results = orchestrator.run_all()
    
    # Save results
    orchestrator.save_results()


def cmd_generate_tables(args):
    """Generate LaTeX tables from results JSON."""
    json_path = Path(args.input)
    
    if not json_path.exists():
        print(f"✗ Results file not found: {json_path}")
        return
    
    with open(json_path, 'r') as f:
        results_data = json.load(f)
    
    results = [ExperimentResult(**r) for r in results_data]
    
    output_path = Path(args.output)
    ResultsAggregator.generate_latex_table(results, output_path)
    print(f"✓ LaTeX table saved to: {output_path}")


# ============================================================================
# SIMDL v1.1 CLI Commands  
# ============================================================================

def cmd_pareto(args):
    """Generate Pareto frontiers from proof.csv files."""
    import glob
    
    # Expand glob patterns
    csv_paths = []
    for pattern in args.proof_csvs:
        csv_paths.extend([Path(p) for p in glob.glob(pattern)])
    
    if not csv_paths:
        print("✗ No proof.csv files found")
        return
    
    # Parse grouping columns
    group_by = args.group_by.split(',') if args.group_by else ['domain', 'agent_id', 'toolset_id']
    
    # Parse lambda values
    lambdas = [float(x.strip()) for x in args.lambdas.split(',')]
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(csv_paths)} CSV files...")
    print(f"Grouping by: {group_by}")
    print(f"Price column: {args.price}")
    print(f"Lambda values: {lambdas}")
    
    ResultsAggregator.generate_pareto_curves(csv_paths, group_by, args.price, lambdas, output_dir)
    print(f"✓ Pareto curves saved to: {output_dir}")


def cmd_ci(args):
    """Compute confidence intervals for metrics."""
    csv_path = Path(args.proof_csv)
    if not csv_path.exists():
        print(f"✗ Proof CSV not found: {csv_path}")
        return
    
    result = ResultsAggregator.compute_confidence_intervals(
        csv_path, args.price, args.lambda_val, args.metric, args.n_boot
    )
    
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Confidence intervals saved to: {output_path}")
    print(f"  {args.metric}: {result['point_estimate']:.4f} [{result['ci_low']:.4f}, {result['ci_high']:.4f}]")


def cmd_plots(args):
    """Generate plots from frontier files."""
    import glob
    
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Pareto plots
    frontier_files = [Path(p) for p in glob.glob(str(Path(args.frontiers) / "frontier_*.csv"))]
    if frontier_files:
        pareto_path = output_dir / "pareto_frontiers.png"
        ResultsAggregator.generate_pareto_plot(frontier_files, pareto_path)
        print(f"✓ Pareto plot saved to: {pareto_path}")
    
    # ROI plots (if proof CSVs available)
    if hasattr(args, 'proof_csvs') and args.proof_csvs:
        csv_paths = [Path(p) for p in glob.glob(args.proof_csvs)]
        if csv_paths:
            roi_path = output_dir / "tool_roi.png"
            ResultsAggregator.generate_tool_roi_plot(csv_paths, args.price, roi_path)
            print(f"✓ ROI plot saved to: {roi_path}")
            
            # Advantage histogram
            advantage_path = output_dir / "advantage_histogram.png"
            ResultsAggregator.generate_advantage_histogram(csv_paths, args.lambda_val, args.price, advantage_path)
            print(f"✓ Advantage histogram saved to: {advantage_path}")


def main():
    parser = argparse.ArgumentParser(
        description="CandleZip Science Tool: Comprehensive Benchmarking Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Initialize configuration
    python science_tool.py init --output benchmark_config.json
    
    # Add domains and files
    python science_tool.py add-domain --name academic --file benchmarks/paper.txt
    python science_tool.py add-domain --name wikipedia --file benchmarks/wiki.txt
    
    # List configured domains
    python science_tool.py list-domains
    
    # Run benchmarks
    python science_tool.py run
    
    # Generate tables from results
    python science_tool.py generate-tables --input results.json --output table.tex
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', required=True)
    
    # Init command
    parser_init = subparsers.add_parser('init', help='Initialize new configuration')
    parser_init.add_argument('--output', default='benchmark_config.json', help='Output configuration file')
    parser_init.set_defaults(func=cmd_init)
    
    # Add-domain command
    parser_add = subparsers.add_parser('add-domain', help='Add domain and file')
    parser_add.add_argument('--config', default='benchmark_config.json', help='Configuration file')
    parser_add.add_argument('--name', required=True, help='Domain name (e.g., academic, wikipedia)')
    parser_add.add_argument('--file', required=True, help='File to add to domain')
    parser_add.set_defaults(func=cmd_add_domain)
    
    # List-domains command
    parser_list = subparsers.add_parser('list-domains', help='List configured domains')
    parser_list.add_argument('--config', default='benchmark_config.json', help='Configuration file')
    parser_list.set_defaults(func=cmd_list_domains)
    
    # Run command
    parser_run = subparsers.add_parser('run', help='Run benchmarks')
    parser_run.add_argument('--config', default='benchmark_config.json', help='Configuration file')
    parser_run.add_argument('--binary', help='Path to CandleZip binary (auto-detect if not specified)')
    parser_run.set_defaults(func=cmd_run)
    
    # Generate-tables command
    parser_gen = subparsers.add_parser('generate-tables', help='Generate LaTeX tables from results')
    parser_gen.add_argument('--input', required=True, help='Results JSON file')
    parser_gen.add_argument('--output', required=True, help='Output LaTeX file')
    parser_gen.set_defaults(func=cmd_generate_tables)
    
    # SIMDL v1.1 Commands
    
    # Pareto command
    parser_pareto = subparsers.add_parser('pareto', help='Generate Pareto frontiers')
    parser_pareto.add_argument('--proof-csvs', required=True, nargs='+', help='proof.csv file patterns')
    parser_pareto.add_argument('--group-by', default='domain,agent_id,toolset_id', help='Grouping columns (comma-separated)')
    parser_pareto.add_argument('--price', required=True, choices=['price_transcript_bits', 'price_pointer_bits', 'agent_duration_ms'], help='Price column')
    parser_pareto.add_argument('--lambdas', default='0,0.25,0.5,1,2,4,8', help='Lambda values (comma-separated)')
    parser_pareto.add_argument('--out-dir', required=True, help='Output directory')
    parser_pareto.set_defaults(func=cmd_pareto)
    
    # CI command
    parser_ci = subparsers.add_parser('ci', help='Compute confidence intervals')
    parser_ci.add_argument('--proof-csv', required=True, help='proof.csv file')
    parser_ci.add_argument('--price', required=True, choices=['price_transcript_bits', 'price_pointer_bits', 'agent_duration_ms'], help='Price column')
    parser_ci.add_argument('--lambda', dest='lambda_val', type=float, default=1.0, help='Lambda value')
    parser_ci.add_argument('--metric', required=True, choices=['net_bits', 'gate_rate', 'mean_advantage'], help='Metric to compute CI for')
    parser_ci.add_argument('--n-boot', type=int, default=2000, help='Number of bootstrap samples')
    parser_ci.add_argument('--out', required=True, help='Output JSON file')
    parser_ci.set_defaults(func=cmd_ci)
    
    # Plots command
    parser_plots = subparsers.add_parser('plots', help='Generate plots')
    parser_plots.add_argument('--frontiers', required=True, help='Directory containing frontier_*.csv files')
    parser_plots.add_argument('--proof-csvs', help='proof.csv file pattern for ROI/advantage plots')
    parser_plots.add_argument('--price', default='price_transcript_bits', choices=['price_transcript_bits', 'price_pointer_bits', 'agent_duration_ms'], help='Price column')
    parser_plots.add_argument('--lambda', dest='lambda_val', type=float, default=1.0, help='Lambda value for advantage histogram')
    parser_plots.add_argument('--out-dir', required=True, help='Output directory')
    parser_plots.set_defaults(func=cmd_plots)
    
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
