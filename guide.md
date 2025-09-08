Agentic Compression-as-Training Arena (KISS, DRY)
### Understanding the Code and Current Implementation

Before diving into fixes, I'll summarize my full understanding of the provided code (from `main.rs` and `Cargo.toml`). I've analyzed it top-to-bottom, including the truncated sections (inferring from context, logs, and typical patterns in similar systems). This is a Rust-based compressor using:

- **Core Compression Mechanism**: Arithmetic coding (AC) with a base-2, 32-bit precision coder for bitstream output. It encodes/decodes tokens predicted by a Llama-based language model (via Candle framework). The model conditions on history (sliding window of tokens, default context=512). Tokens come from a HuggingFace tokenizer. Repriming (recomputing hidden states) occurs periodically to handle long sequences without OOM.
  
- **Container Format**: A binary header (v2) with magic number, version, fixed fields (e.g., BOS token ID, token count, hashes for model/tokenizer/original data), and variable-length fields (e.g., model file repr). Payload is the AC-compressed bitstream. Hashes (BLAKE3 truncated to 16 bytes) ensure integrity.

- **Model Loading**: Downloads from HF Hub if not local (default: smollm-135M-instruct-v2). Supports sharded models via index.json. Device: CUDA if available, else CPU.

- **Encoding Process**:
  - Tokenize input text (add BOS).
  - Initialize model session (LmSession wraps LlamaModel, Cache, etc.).
  - For each token: Get logits, compute softmax PDF (with floor to avoid zero-prob), find cumulative bounds, encode interval via AC.
  - Reprime safely if context exceeded or interval hit.
  - Progress bar for UX.

- **Decoding Process**:
  - Read header, verify hashes (unless --force-mismatch).
  - Load same model/tokenizer.
  - Initialize AC decoder on payload.
  - Start with BOS, decode symbols using PDF from model logits, advance model step-by-step.
  - Detokenize output (skip BOS).

- **Agent Mode (--agent)**:
  - Activated via flag; sets reserved_flags &1 in header.
  - Processes in batches (default agent_batch_bytes=512 bytes, ~100-200 tokens).
  - For each batch: 
    - Generate query_text = full decoded_text so far (accumulated output).
    - Perform "research" via `perform_agent_research`: 
      - If arXiv ID detected (via `extract_arxiv_id`, likely a regex like r"\d{4}\.\d{5}(v\d)?"), directly download paper.
      - Else: Extract top-3 freq keywords (alphanumeric, lowercase, sorted by freq then lex).
      - Use MCP client to call tools via gateway (e.g., Docker-based server).
      - Tools selected by predicates (e.g., ["duck", "search"] for DuckDuckGo, ["wiki"] for Wikipedia, ["arxiv", "search"] for arXiv search).
      - For arXiv: Search, then download (inferred from truncation: parses search results for top paper_id, downloads it).
      - Concat results as ghost_text.
    - Estimate entropy: Baseline (reprime on history, sum -log2(p) for batch tokens) vs. ghost (reprime on history + ghost_tokens, same sum). Add 1-bit overhead for gate.
    - If ghost_bits + overhead < baseline_bits (est_gain >0), use ghost (encode gate=1), else skip (gate=0).
    - Encode gate bit, then encode batch tokens using the chosen priming.
  - MCP Client: JSON-RPC over stdin/stdout to external servers (e.g., Docker). Caches tools/results. Deterministic calls.
  - Training Data: Logs to agent_data.jsonl (batch stats, tool calls, overlap_score = keyword Jaccard between batch and history).
  - Decode replicates: Reads gate bit, if 1, repeats exact research (same query, tools) to get identical ghost.

- **MCP (Model Context Protocol)**: From code and quick search (it's a speculative/emerging protocol for tool-augmented LLMs, similar to LangChain but RPC-based). Allows listing/calling tools with schemas. Here, used via gateway for web/arXiv tools (search, download_paper, etc.). Responses parsed as text/content.

- **Dependencies** (from Cargo.toml): Standard (anyhow, clap, etc.) + Candle for ML + Tokio for async MCP + tokenizers + hf-hub.

- **Issues from Logs (agent_data.jsonl)**:
  - **Vague Queries**: Early batches use "general knowledge" or common words ("the of a", "the of and", "the of 0"). Leads to irrelevant fetches (wrong paper_ids like "2412.04948v1", "2111.07619v1").
  - **Tool Redundancy**: Often repeats search_papers twice, wastes calls.
  - **Hit-or-Miss Detection**: Sometimes fetches correct "2306.04050v2" (likely when arXiv ID appears in query_text), but ghost_len varies (120 to 29449 bytes—perhaps partial/full downloads).
  - **Gating Failures**: used_ghost=false most times, est_bits negative (e.g., -585). Even with correct ghost, gain negative until late batches (when history aligns better?).
  - **Overlap Decay**: High early (batch keywords match history), low late (dilution).
  - **Repetition**: Patterns repeat across runs, showing determinism but stuck in suboptimal searches.
  - **Overall**: Minimal gain (0.964 -> 0.940 BPB) because ghost rarely used, and when used, not optimally conditioned (raw concat causes mismatch, as explained below).

The paradigm: This is a proof-of-concept for agentic compression, where tools provide "ghost data" (side info) to reduce conditional entropy. It approximates Solomonoff by exploring hypotheses (via tools) that explain data. Decoder replicates fetches, so no info leak. But current rule-based agent is dumb; needs to be LM-driven for intelligence.

### Key Technical Problems and Why It's Failing

1. **Query Generation is Too Crude and Global**:
   - Uses full decoded_text for query_text, then top-3 freq words. For large files, this yields stopwords ("the of a"), leading to useless searches.
   - arXiv detection (extract_arxiv_id) works only if ID in full text, but early batches may miss it if ID appears later.
   - Result: Irrelevant ghost (wrong papers), no entropy reduction.

2. **Tool Selection and Execution is Hard-Coded and Inefficient**:
   - Predicates/static sequences (Duck -> Wiki -> arXiv search -> download first hit). No adaptation based on content.
   - Redundant calls (e.g., search_papers twice).
   - No multi-step reasoning: For arXiv, it searches but picks arbitrary ID (likely top result), not verifying relevance.
   - Cache helps, but doesn't prevent bad queries.
   - MCP underused: Lists tools but doesn't dynamically choose based on context.

3. **Ghost Conditioning is Naive (Raw Concatenation)**:
   - Primes with history_tokens + ghost_tokens.
   - When ghost = full paper (exact match), this becomes prefix + full_paper. Then predicting next batch (e.g., middle of paper) after that position expects post-paper content, not mid-paper. Logits assign low prob to actual tokens → ghost_bits high → negative gain → skip.
   - Even if relevant (e.g., summary), no separator/prompt; model treats as continuation, confusing narrative.
   - Large ghosts (e.g., 139k bytes) may exceed context (default 512), though code truncates to effective_context.
   - Result: Even perfect ghost often hurts entropy, as seen in negative est_bits.

4. **Gating/Entropy Estimation is Conservative and Misaligned**:
   - est_gain = baseline_bits - ghost_bits - 1 (overhead).
   - But if concat mismatches, ghost_bits > baseline_bits → negative.
   - Overhead=1 bit is arbitrary; for large gains (e.g., 490 bits in logs), it's fine, but ignores fetch cost (though offline).
   - Computed per-batch, but fetches use full history query → late batches benefit more (full text has ID → correct fetch → positive gain when history long enough to align).
   - No iteration: Fetches once per batch; no refinement if ghost bad.

5. **Batch Size Too Small (512 bytes)**:
   - Small batches → frequent fetches/estimations → overhead amplifies.
   - Early small history → poor queries → wasted.
   - User request: Bump to 2048 for better context per decision.

6. **Lack of True Agentic Behavior**:
   - Rule-based, not LM-driven. Doesn't use the model for decisions (e.g., "what tool would help predict next?").
   - Not general: Hard-coded for web/arXiv. Adding tools (via MCP) won't auto-adapt.
   - No evaluation loop: Doesn't assess fetched ghost's usefulness beyond entropy (e.g., no similarity check).
   - Determinism ok, but rigidity prevents approximating Solomonoff (exploring diverse hypotheses).

7. **Decode Symmetry**: Fine, but if agent smarter (LM-based), ensure greedy decode for determinism.

Overall Failure in Benchmark: For ArXiv file with early citation, early batches miss ID (small history), fetch junk, skip. Later: fetches correct paper, but raw concat causes entropy spike (mismatch) until history ~ half paper (alignment?). Gains only in last batches → minimal overall BPB improvement. Not >100x because ghost not used as "source" (no copying/matching logic).

### Fixes to Achieve Agentic Compression Framework

Aim: Transform into a general framework where any MCP tools can be added (e.g., code exec, DB query), and the LM acts as agent to select/call them for entropy reduction. Agnostic to tools/files—focus on context-aware decisions. No specifics to ArXiv/current benchmark; benchmark just exposes flaws. Target: Massive ratios when side-info available (e.g., web replicas), via better conditioning. Preserve determinism (greedy LM, fixed seeds).

1. **Change Default Batch Size to 2048**:
   - In Cli: Change #[arg(long, default_value = "512")] to "2048".
   - Rationale: Larger batches amortize fetch/estimation cost, give more local context for queries. Logs show small batches lead to fragmented decisions.

2. **Improve Query Generation (Local, Smarter)**:
   - Change query_text = last 2048 bytes of decoded_text (or current batch + last 1024).
   - Enhance extract_keywords: Use top-5, filter stopwords (e.g., skip "the", "of", "a", "and"). Add phrases (n-grams) or entities (but keep simple, no extra deps).
   - For extract_arxiv_id: Scan only recent 4096 chars (likely where citations are).
   - Add general regex for identifiers (e.g., DOIs, URLs) to trigger relevant tools.

3. **Make Tool Selection Dynamic and LM-Driven (True Agentic)**:
   - In perform_agent_research: List all tools via client.list_tools().
   - Use the LM itself (same session/model) for decisions:
     - Prompt: System: "You are an entropy-reducing agent. Given text history, suggest up to 2 tool calls to fetch info helping predict continuation. Tools: [list with names/descriptions/schemas]. Output JSON: [{'name': '..', 'arguments': {...}}]"
     - User: "History: [last 1024 chars of decoded_text]\nSuggest tools to reduce uncertainty."
     - Greedy decode (temp=0) up to 256 tokens, parse JSON.
     - Call suggested tools (up to 2, parallel if possible), concat results as ghost_text.
   - Why: LM decides based on content (e.g., sees citation → calls arXiv with ID). Generalizes to any tools (e.g., if "math" tool added, uses for equations).
   - Overhead: Extra LM calls, but offline ok. Deterministic via greedy.

4. **Fix Ghost Conditioning with Prompt Wrapping**:
   - Don't raw concat. Instead: prime_ids = tokenize( "Additional research context: \n" + ghost_text + "\n\nOriginal text continuation: " ) + history_tokens
   - Use instruct format (since smollm-instruct): BOS + [INST] ghost as system info [/INST] + history.
   - If ghost large, summarize with LM (greedy: "Summarize for prediction: [ghost]" → short version).
   - Why: Treats ghost as side info, not continuation. Model (instruct-tuned) can reference it without position mismatch. For exact content: Prompt like "The text is from this source: [ghost]\nRepeat/continue it:" → high prob for matching tokens.
   - Test: In estimation, use same prompt for ghost priming.

5. **Refine Gating/Entropy Estimation**:
   - Adaptive overhead: 1 + log2(num_tools) bits (for potential choice encoding, but since deterministic, maybe 0).
   - Add similarity bonus: If ghost overlaps batch (e.g., Jaccard >0.5), boost gain by fixed bits (encourage matches).
   - Iterate if needed: If first ghost gives negative gain, try one more tool call (LM-suggested alternative).
   - Threshold: Use if gain > -10 (tolerate small losses for potential).
   - Why: Current conservative; allows using near-exact ghost despite minor mismatch.

6. **General Enhancements for Framework**:
   - **Tool Agnosticism**: Always list/call via MCP. No hard-coded predicates—LM picks.
   - **Multi-Round Agent**: Allow 1-2 rounds: After first fetches, LM evaluates: "Does this help? Need more?" → refine.
   - **Training Data Enrichment**: In jsonl, add "actual_gain_bits" (post-encode, compare compressed size with/without).
   - **Error Handling**: If tool fails (empty), skip without crash; log as ok=false.
   - **Context Management**: If history + ghost > context, truncate ghost first (least important).
   - **Decode Parity**: Same prompts/LM calls in decode.
   - **No Specificity**: Avoid ArXiv hard-codes; LM learns via prompt (e.g., tools include "arxiv_download").

With these, it becomes a true agentic framework: LM-as-agent explores tools to minimize entropy, generating pretraining data for future agents. For benchmark, LM will spot ID early, call download, condition properly → use ghost often → near-zero BPB (model predicts exact with high confidence). Test iteratively; start with prompt wrapping + local queries for quick wins.