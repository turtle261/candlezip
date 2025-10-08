# CandleZip: Agentic Compression for Auditable Intelligence

CandleZip exists to turn the well-known statement "compression = intelligence" into a falsifiable
engineering instrument. The project frames every tool call as a priced entropy sink:
an agent should only spend resources when those resources route chaos(entropy) out of
the data stream. The result is a research-grade benchmark directly measures intelligence, and emits an integer-bit
signal you can trust, cite, and extend. Measure something that compresses **everything**, and that definitionally is general intelligence!

DOI: [10.5281/zenodo.17282860](https://doi.org/10.5281/zenodo.17282860)

---

## 1. Why CandleZip Exists

1. **Compression as intelligence, made concrete.** CandleZip operationalizes the
   long-standing view ([Mahoney](https://mattmahoney.net/dc/dce.html#Section_14), [Hutter](https://www.hutter1.net/ai/uaibook.htm#oneline), MDL) that predictive skill equals entropy
   reduction. We expand it agentically: an intelligent policy routes entropy into
   controllable sinks (tools, retrieval corpora, code interpreters) and earns credit
   only when the routed bits beat their price. 
2. **A small, falsifiable law.** The Unified Efficiency Principle defines intelligence
   as entropy reduced per priced resource. CandleZip instantiates that law with an
   entropy coder so that every gain is logged in whole bits, complete with
   shipped-overhead accounting. If the agent cheats—by leaking oracles or skipping
   decode-safety—the ledger will expose it.
3. **A bridge between theory and practice.** The same audit trail applies to LLMs,
   classical compressors, and humans (via a UI agent). Compression becomes a
   substrate-agnostic intelligence instrument rather than an abstract metaphor.
---

## 2. What is Intelligence, and how do you measure it?

The concept of intelligence = compression is not new, it extends work from early information theorists, and in particular Mahoney.
However, practical measures of intelligence **had** yet to be established on this principle. 

We use “definition” in an operational, information-theoretic sense: an agent is
intelligent to the extent that, under decode-safety and priced side-information, its
actions reduce the expected codelength of its own sensor/working history per unit
cost. This criterion is intended as necessary-for-competence and empirically useful
—not metaphysically sufficient—so non-adaptive dissipative systems are not by this
definition considered significantly Intelligent (example: Fridge).

Therefore, Intelligence can be measured by lossless compression, where reduced file
size is quite literally entropy reduction (more precisely, relocation).

This is substrate Agnostic--human intelligence can be measured using Candlezip, by
using a UI python script, rather than the LLM Agent script provided--this is WIP.
Feel free to implement this and PR.

**How CandleZip measures it.**
- We track **gross entropy reduction** (\widehat{ER}): the integer-bit drop between
  the baseline stream and the stream conditioned on accepted hints.
- We subtract **shipped overhead** (gate records, hint payloads, headers) to obtain
  **net savings**. Gross and net are both published so audits can price resources
  differently.
- We report **ROI** as bits saved per priced resource (time, bytes, $), yielding an
  operational intelligence score consistent with the representation theorem in the
  accompanying preprint.

---

## 3. System Overview — From Philosophy to Implementation

CandleZip implements a sink-inclusive minimum description length workflow (SIMDL).
The system treats tools as priced observation channels (entropy sinks) and measures
the agent's effectiveness by how many bits the agent saves per unit cost. Key
concepts are:

- **Entropy routing.** Each chunk is encoded twice (baseline vs. tool-conditioned).
  The gate keeps a hint if its measured gross drop exceeds its priced cost.
- **Decode-safe accounting.** All information required for deterministic decode is
  either cached or shipped and debited explicitly; nothing is "free" just because a
  tool produced it.
- **Capacity–efficiency decomposition.** Capacity is the channel-side
  mutual-information ceiling (ideal limit under optimal coding); efficiency is the
  agent's realized fraction under the observed coder and budget.
- **Deterministic proofs.** Every accepted hint is hashed, logged, and replayable so
  third parties can reproduce the exact bitstream.

### Key features

- Deterministic, replayable encoding and decoding with cached agent outputs for
  exact verification.
- Gross vs. net accounting: gross bitstream reductions are reported alongside shipped
  costs so users can evaluate net savings.
- Gate-based hint acceptance: the system accepts side-information only when the
  expected bit savings exceed the priced cost (including shipped bits and any
  exogenous costs).
- Structured logging and per-chunk CSV output for offline analysis and reproducible
  audits.
- Tools for Pareto analysis and statistical validation (bootstrap confidence
  intervals, paired tests).

---

## 4. Reproducibility Playbook

1. **Build once, run anywhere.**
   ```powershell
   # Windows PowerShell example
   ./build.ps1
   ```
2. **Deterministic self-test.**
   ```bash
   ./target/release/candlezip.exe \
     --backend smollm \
     --agent \
     --scan \
     --scan-lookahead 512 \
     --context 512 \
     --reprime-interval 512 \
     --scan-agent-script agent/agent_v2.py \
     --scan_max_steps 12 \
     self-test benchmarks/your_file.txt
   ```
3. **Compress & decompress with replay.**
   ```bash
   ./target/release/candlezip.exe compress \
     --backend smollm \
     --agent \
     --scan \
     --scan-lookahead 512 \
     --context 512 \
     --reprime-interval 512 \
     --scan-agent-script agent/agent_v2.py \
     --scan_max_steps 12 \
     input.txt output.canz
    # Use --reuse instead, if your Agent is deterministic. In that case, no cache is needed to decompress!
   ./target/release/candlezip.exe decompress \
     --backend smollm \
     --agent \
     --reuse-scan-dir dir/to/compressed/run \
     --scan-agent-script agent/agent_v2.py \
     output.canz decoded.txt
   ```

### What the artifact ships

- Encoded files with gate records (compact per-chunk metadata) and optional shipped
  hint descriptions.
- `proof.csv` files with per-chunk baseline/conditioned bits, bits-saved, gate
  decisions, pricing fields, and timing columns.
- Watchdog logs plus cached agent outputs for deterministic replay and audit.

---

## 5. Results Snapshot — 300 s Budget, No Memory Cache - Gemini 2.5 Pro

A 300 s per-run budget with no persistent memory was evaluated on Canterbury
subsets. The table summarizes the accompanying `proof.csv` ledgers in
`results_300s_nomem/`.

| Dataset (run) | Baseline bits | Conditioned bits | Bits saved | % saved | Gate accept rate | Avg agent latency (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `alice29.txt` (`alice29_selftest_smollm_20251003_043405`) | 140,737.10 | 108,941.08 | 31,796.02 | 22.59% | 53.09% | 60,613.19 |
| `asyoulik.txt` (`asyoulik_selftest_smollm_20251003_121454`) | 162,104.20 | 138,540.06 | 23,564.14 | 14.54% | 38.96% | 48,325.10 |

- **Instant read:** CandleZip’s gate accepted 43 of 81 chunks on `alice29` and 30 of
  77 on `asyoulik`, yielding double-digit percentage savings despite priced
  overheads. Every number above is an integer-bit measurement aggregated from the
  released `proof.csv` logs—no illustrative placeholders.
- **Dig deeper:** Inspect the per-chunk CSVs in `results_300s_nomem/<run>/proof.csv`
  to audit individual decisions, shipped bits, and timing.
- **Note:** This is quite expensive--benchmarking an AI model... Feel free to submit results yourselves or offer to support it if so kind! Secondly, we use Gemini 2.5 Pro as the reasoning LLM, and 2.5 Flash as the base LLM, in preliminary tests, this results in near-identical performance.

---

## 6. Trust Signals & Extensibility

- **Determinism:** Temperature 0.0, fixed seeds, cached agent outputs, replayable
  decode with no network calls.
- **Decode-safe accounting:** Gate records (`GateRecordV2`), tool descriptors, and
  hint payloads are all priced. Gross ≥ 0 by construction; net may be negative if you
  overspend the budget.
- **Human parity:** Swap in a human-facing Python UI to evaluate human analysts under
  identical budgets—no code changes required in the Rust core.
- **Research hooks:** Budget sweeps, ablations, and bootstrap tests are scriptable via
  the logged CSV schema (see `results_300s_nomem/.../proof.csv`).

---

## 7. License & Contact

This README and related documentation are licensed under Creative Commons
Attribution 4.0 International (CC BY 4.0).

CandleZip source code is licensed under GNU General Public License v3.0 (GPL-3.0).

See the repository root for licensing and contribution information. For questions,
open an issue, PR, or email me.

## Potential Next Steps
- Extending to be multimodal, vision first?
- Run RL on the information reward signal this returns to directly optimize for Intelligence in Agentic, tool using systems--multimodal too? Shatter records?
