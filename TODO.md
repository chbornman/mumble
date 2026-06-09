# TODO

Open roadmap items. Each has enough context that picking one up later doesn't
require re-deriving the motivation. Everything from the original v1 roadmap
(LLM post-processing, per-app context, preset modes, calibration, clipboard
threshold, voice command mode, stream dedup) has shipped — see ARCHITECTURE.md
for how those pieces fit together today.

---

## 1. Internal naming cleanup (`whisper_*` → `mumble_*`)

Cosmetic but visible: the daemon module (`whisper_daemon.py`), IPC socket
(`/tmp/whisper_daemon.sock`), flag files (`/tmp/whisper_recording`,
`/tmp/whisper_streaming`), log file, and logger channel all predate the rename
to mumble. Renaming is mechanical but breaks deployed setups (keybinds, waybar,
anything watching the flag paths), so it should land in one commit with a
config-driven migration note in the README.

## 2. Streaming + LLM post-processing combined

Architecturally awkward — a cleanup pass behind live streaming pushes pipeline
latency from <1s to 5-10s (FreeFlow's FAQ documents the same finding), and no
surveyed OSS project ships it. Revisit only with a clear UX design (e.g.
preview-before-commit). Until then, LLM cleanup stays a press-to-talk feature.

## 3. Hosted LLM cleanup endpoint

`[llm_postprocess]` is fully wired but ships disabled: it needs an OpenAI-style
endpoint (llama.cpp server, ollama) and keeping a model resident costs VRAM the
ASR models also want. A sensible setup hosts the cleanup model on a second
machine or loads it on demand (`toggle_llm.sh` exists for exactly this). Document
a recommended single-machine recipe once one proves out.

---

## Parked / research-only

Surveyed in the OSS audit but deferred, with reason.

- **LLM-generated activity summary** (FreeFlow's pre-flight screenshot-assisted
  LLM → 2-sentence intent description). Adds a second LLM call per turn; per-app
  context injection covers ~80% of the benefit cheaper. Revisit only if per-app
  context proves insufficient.
- **Keyring-based API key storage** (Python `keyring`). Add only if cloud LLM
  endpoints ever replace local.
- **Continuous-mode API safety guards** (`allow_continuous_api`, timeouts).
  Only relevant if continuous dictation + cloud endpoints both exist.
- **Per-language Whisper `--prompt`** (per-language prompt dicts). English-only
  for now.
