# TODO

Roadmap items, with enough context that picking one up later doesn't require re-deriving the motivation.

---

## 1. LLM post-processing on top of Whisper (CLI mode)

**Goal:** Add an optional local-LLM cleanup pass after Whisper transcription, scoped to CLI mode. Target quality: "decently close to Wispr Flow" without leaving the machine.

**Scope for now:** CLI mode only (press-to-talk, release, get cleaned text). Streaming + LLM is **explicitly out of scope** — it's architecturally awkward (cleanup needs a finished utterance; streaming emits partials before a thought closes) and should be revisited after the CLI flow is solid.

**What cleanup should do:**
- Remove fillers: "um", "uh", "like", "you know"
- Collapse backtracks: "meet Tuesday — wait, Wednesday" → "meet Wednesday"
- Fix repeats/stutters: "I, I think" → "I think"
- Restore punctuation and capitalization beyond Whisper's baseline
- Expand obvious acronyms where context is unambiguous
- **NOT** rewrite for style, tone, or concision — that's a separate mode

**What cleanup must NOT do:**
- Invent content
- "Correct" domain terms that are already right (vocab.txt list should be passed to the LLM as "preserve exactly")
- Change meaning. Prompt should instruct: *minimum edit distance; preserve wording where possible.*

**Architecture sketch:**
- New `[llm_postprocess]` section in `config.toml`: `enabled`, `backend` (`llama.cpp` / `ollama` / `vllm`), `endpoint`, `model`, `prompt_template`, `max_tokens`, `temperature` (default 0.0).
- New method `transcription.postprocess(text, vocab)` called after `clean_text` in `whisper_daemon.py:407`.
- Waybar toggle to enable/disable per-session — the first time it mangles a command, you'll want a one-keypress bypass.
- Log raw Whisper output + cleaned output to a diff log for eyeballing quality over the first few weeks.

**Local model candidates (RTX 5080, 16 GB):**
- **Qwen3-4B-Instruct** (Q5_K_M, ~3 GB VRAM) — fast, strong at constrained-edit tasks
- **Phi-4-mini** (3.8B) — small, instruction-tuned, low latency
- **Llama-3.1-8B-Instruct** (Q4_K_M, ~5 GB) — well-understood baseline
- **Hermes-3-Llama-3.1-8B** — cited repeatedly in the literature for this exact use case (Whisper → local LLM cleanup)

Run via `llama.cpp` server on a local port — mumble already uses whisper.cpp subprocesses, same pattern applies.

**What the research says (April 2026):**
- Hybrid ASR→LLM cleanup is standard practice, not experimental. OpenAI's own cookbook documents the pattern.
- "Whisper: Courtside Edition" (multi-agent LLM post-processing) reports ~17% relative WER reduction on domain-specific transcription with no model retraining.
- Multi-model reconciliation (several ASR outputs + LLM arbitration) reports up to 40% fewer critical errors in production — probably overkill here, but worth knowing.
- Documented failure modes across the literature: hallucinated edits, over-correction of proper nouns, inconsistent performance on short utterances. Constrained prompts + low temperature + a diff log to catch regressions are the standard mitigations.
- **Notable trend to watch, not adopt yet:** native audio-in LLMs (Qwen2.5-Omni, Gemini, GPT-4o-audio) bypass ASR entirely for some downstream tasks. Different architecture, different product. Park it.

**References:**
- [OpenAI Cookbook — Enhancing Whisper transcriptions: pre- & post-processing](https://cookbook.openai.com/examples/whisper_processing_guide)
- [Whisper: Courtside Edition — LLM-driven context generation (2025)](https://www.researchgate.net/publication/401133518_Whisper_Courtside_Edition_Enhancing_ASR_Performance_Through_LLM-Driven_Context_Generation)
- [ASR Error Correction using LLMs (Cambridge)](https://www.repository.cam.ac.uk/bitstreams/55e62442-b4f4-4212-bbd1-a6e24d427dc1/download)
- [Whisper-LM: Improving ASR Models with Language Models (arXiv 2503.23542)](https://arxiv.org/html/2503.23542v1)
- [Build with GenAI: Whisper + local LLM (Medium)](https://medium.com/design-bootcamp/build-with-genai-turn-rambling-into-writing-with-whisper-and-local-llm-394e8dd5b83f)
- [Using Whisper + Mixtral8x7B via llama.cpp](https://brandolosaria.medium.com/using-openai-whisper-and-mixtral8x7b-to-transcribe-and-correct-grammar-from-videos-da1d243fc157)

---

## 2. Calibration command — tune stream/VAD/dedup settings to my voice

**Goal:** A one-shot calibration subcommand that records a known passage read at my natural cadence, sweeps the relevant config knobs, and picks values that minimize dupes and dropped words for *me specifically*. One-time setup per machine.

**Why:** Defaults like `vad_threshold = 0.6`, `keep = 200`, `min_overlap_chars = 20`, `drift_reset_threshold = 2` were picked for some imagined average voice. The frustration with stream-mode inconsistency is partly that those are wrong for me. Calibration is program-side tuning that respects my natural speaking rhythm rather than forcing me to change how I talk.

**Interface sketch:**
```bash
mumble calibrate              # Interactive: prompts to read passage, records, sweeps
mumble calibrate --passage benchmarks/passages/technical.txt
mumble calibrate --apply      # Write tuned values to config.local.toml
mumble calibrate --report     # Show results without applying
```

**What to sweep:**
- `streaming.vad_threshold` — sensitivity of voice activity detection
- `streaming.keep` — buffer overlap between emits
- `streaming.min_overlap_chars`, `overlap_step`, `fallback_suffix_length` — deduplicator heuristics
- `streaming.drift_reset_threshold` — how eagerly to reset committed text on fallback

**How to score:**
- Input: known ground-truth text + recorded audio of me reading it
- Metric: WER against ground truth, **plus** a dupe-rate metric (substring-level: count emitted tokens that aren't in ground truth but duplicate an earlier emitted token). Pure WER won't catch the "Inconsistent. Inconsistent." failure because both are technically in the text.
- Output: a small grid-search report — best config, second-best, sensitivity of each knob. Reuses `benchmark.py` infrastructure where possible.

**Practical notes:**
- Benchmark corpus already exists in `benchmarks/` — extend it rather than create a new one.
- Passage should be ~2-3 minutes with deliberate clause-boundary pauses *and* a few mid-phrase hesitations, so the tuning covers the hard case.
- Write tuned values to `config.local.toml` (not `config.toml`) so they stay per-machine.

---

## 3. Stream-mode dedup fix (separate from calibration)

**Goal:** Replace `StreamDeduplicator`'s character-overlap heuristics in `whisper_daemon.py:51` with logic that handles the observed failure mode: pause mid-phrase → VAD emits → user resumes → dupe gets typed.

**Why separate from #2:** Calibration tunes the *existing* heuristics. This item replaces them when even tuned values are inadequate. Do calibration first — if it solves the pain, this item may downgrade to minor cleanup.

**Approach to consider:**
- Word-level diff instead of character-level overlap matching (more robust to Whisper slightly rewording a token between blocks).
- Explicit "immediate-repeat suppression": if the new emit starts with an exact or near-exact repetition of the last N words already typed, drop it.
- Constrain the strategy-3 fallback (`_extract_last_sentence`) — currently too eager to re-emit.

**Out of scope for first pass:** streaming + LLM cleanup combined. Architecturally awkward (see item #1). Revisit only after CLI-mode LLM is working and stream dedup is stable.
