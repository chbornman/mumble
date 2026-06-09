# Architecture

How mumble is put together: the pieces, the interfaces between them, and the
seams where you can swap or extend behavior. If you want to add an ASR engine,
change how text gets injected, or wire up a new editor/compositor, this is the
map. For the open roadmap, see [TODO.md](TODO.md).

## Design goals

- **One product, modular inside.** A single branded `mumble` you install once.
  Swappable ASR backends, optional LLM cleanup, pluggable text injection — all
  behind small, documented interfaces so you can riff on one part without
  understanding the whole.
- **Config-driven, machine-portable.** Behavior lives in `config.toml`, with
  per-machine overrides in `config.local.toml`. Clone → install → it adapts to
  the hardware. No code edits to move between machines.
- **No magic, no false assumptions.** The installer does only what's reliable on
  any distro and *reports* the rest; the README is the source of truth;
  `mumble doctor` verifies. We never silently run your package manager or edit
  your compositor config.

## The picture

```
   ┌─────────────┐     IPC (unix socket, text verbs)      ┌──────────────────────┐
   │ keybind /   │ ─────────────────────────────────────▶ │     mumble daemon    │
   │ mumble CLI  │   TOGGLE · STREAM · STATUS · SET_MODE  │  (whisper_daemon.py) │
   └─────────────┘                                        └──────────┬───────────┘
        ▲                                                            │
        │ you bind these in                       selects + drives   │
        │ Hyprland/Sway/GNOME/…                                      ▼
        │                                            ┌───────────────────────────┐
        │                                            │      ASR backend          │
        │                                            │  (one of, per config)     │
        │                                            │  • whisper-cli            │
        │                                            │  • whisper-server        │
        │                                            │  • whisper-stream (legacy)│
        │                                            │  • nemotron-streaming ──────▶ mumble-stt
        │                                            └─────────────┬─────────────┘   sidecar
        │                                                          │ text         (own venv +
        │                                       ┌──────────────────▼──────────────┐  service)
        │                                       │ optional LLM cleanup (off by    │
        │                                       │ default; any OpenAI-compatible  │
        │                                       │ endpoint)                       │
        │                                       └──────────────────┬──────────────┘
        │                                                          │ final text
        │                                            ┌─────────────▼─────────────┐
        └──────────────── feedback ◀──────────────── │   text injection          │
            (sounds, notify, waybar)                 │ wtype | xdotool | ydotool │
                                                     │ (+ clipboard-paste path)  │
                                                     └───────────────────────────┘
```

## Components

### 1. The `mumble` CLI and keybind clients

The [`mumble`](mumble) script is the user-facing client: `mumble
toggle|stream|status|mode|doctor`. The verbs are tiny IPC sends to the daemon's
unix socket; `doctor` is a read-only environment preflight (injector, clipboard
tools, whisper binaries, model, services, sockets). The `toggle_*.sh` scripts
are kept as thin back-compat wrappers over the CLI — bind a key to either. This
is the whole keybind surface: mumble never grabs keys itself, because every
compositor does it differently.

### 2. The daemon (`whisper_daemon.py`) — the spine

A long-lived systemd **user** service (`mumble.service`). Owns:
- the IPC socket and the command verbs (`handle_command`),
- recording (press-to-talk) and streaming session lifecycle — *both* streaming
  engines run daemon-owned: the legacy `whisper-stream` pipeline as a managed
  subprocess, the Nemotron engine as a socket client to the sidecar,
- backend selection and the transcribe → cleanup → inject pipeline,
- user feedback (sounds, `notify-send`, waybar state).

**IPC wire format** (newline/whitespace tolerant, backward compatible):
```
TOGGLE | START | STOP | STATUS | STREAM           # core verbs
START mode=commit                                 # per-turn preset override
SET_MODE email | SET_MODE none                    # session default preset
COMMAND                                           # voice-command-on-selection turn
```
Responses are short uppercase tokens (`RECORDING`, `READY`, `STOPPED`,
`MODE_SET <m>`, `UNKNOWN_COMMAND`, …) so any client can parse them.

### 3. ASR backends — the swap point

The core extension seam. A backend turns audio into text; the daemon doesn't
care how.

| Capability | press-to-talk (batch) | streaming (incremental) |
|---|---|---|
| `whisper-cli` | ✅ loads model per call | — |
| `whisper-server` | ✅ persistent HTTP | — |
| `whisper-stream` (legacy) | — | ✅ buffered + dedup heuristics |
| `nemotron-streaming` | — | ✅ true cache-aware RNNT, native P&C |

- **Batch backend** = "give me a WAV, return text" — selected by
  `[backend].type` + `[daemon].mode`.
- **Streaming backend** = "here's a mic stream, emit partial + final segments
  as you go" — selected by `[backend].streaming_backend`.

**Why two streaming engines?** `whisper-stream` approximates streaming with a
rolling fill-then-dump buffer, which re-transcribes overlapping audio — that's
why `stream_dedup.py` / `word_dedup.py` exist. The Nemotron model
(`nvidia/nemotron-speech-streaming-en-0.6b`, cache-aware FastConformer-RNNT) is
purpose-built for streaming: stable incremental tokens, native punctuation +
capitalization, sub-second end-of-utterance — so the entire dedup layer is
unnecessary on that path. whisper-stream is kept because it runs anywhere
whisper.cpp does (CPU/Vulkan, no CUDA), and as the fallback
(`[nemotron].legacy_fallback`) when the sidecar is unavailable.

### 4. The Nemotron sidecar (`mumble_stt/`)

Nemotron runs on **NeMo + PyTorch**, which is multi-GB and pins its own CUDA
torch — it **cannot** share mumble's lightweight venv. So it runs as its own
long-lived process (`mumble-stt.service`) in a dedicated venv
(see [`mumble_stt/requirements.txt`](mumble_stt/requirements.txt)), holding the
model resident and speaking a small framed protocol over a unix socket. The
daemon's `nemotron-streaming` backend
([`backends/nemotron_streaming.py`](backends/nemotron_streaming.py)) is a thin
client to it.

It is still **part of mumble**: same repo, same installer (opt-in extra,
because of its weight), checked by `mumble doctor`. "Sidecar" is a deployment
detail, not a separate project — it also means the heavy ASR could run on a
different machine later without touching the daemon.

| File | Role |
|---|---|
| `mumble_stt/server.py` | Entrypoint. Opens the socket, accepts one session at a time, owns the asyncio loop + lifecycle. What the systemd unit runs. |
| `mumble_stt/engine.py` | NeMo wrapper. Holds the model resident, owns cache-aware streaming state: `open_utterance()` / `feed(pcm)` / `finalize()`. **The only module that imports torch/NeMo.** |
| `mumble_stt/protocol.py` | Shared, pure-stdlib wire contract (framing + message builders/readers). Imported by BOTH sides. |
| `mumble_stt/config.py` | Reads the `[mumble_stt]` table from the same `config.toml`. No second config file. |
| `mumble_stt/__main__.py` | `python -m mumble_stt` alias for `server.py`. |

Run it by hand for debugging (from the repo root, heavy venv):

```sh
.venv-stt/bin/python -m mumble_stt --config config.toml
```

Self-test without a GPU or the heavy venv (config + protocol round-trips only):

```sh
python mumble_stt/server.py --selftest
```

Smoke-test against a WAV, with the sidecar running:

```sh
python -m backends.nemotron_streaming "$XDG_RUNTIME_DIR/mumble-stt.sock" path/to/test.wav
```

### 5. Sidecar wire protocol

The contract between the sidecar (server) and the daemon's backend (client).
[`mumble_stt/protocol.py`](mumble_stt/protocol.py) is the machine source of
truth.

**Transport.** A `SOCK_STREAM` unix socket at `$XDG_RUNTIME_DIR/mumble-stt.sock`
(configurable via `[mumble_stt].socket_path`). The sidecar binds/listens,
unlinks a stale socket on start, and `chmod 0600`s it. **Exactly one session at
a time** — a second connection gets `ERROR{code="busy", fatal=true}` and the
socket closes.

**Framing.** Every frame, both directions:

```
byte 0      : TYPE   (uint8)
bytes 1..4  : LEN    (uint32, big-endian) = PAYLOAD length in bytes
bytes 5..   : PAYLOAD
```

Length prefixing (not newline framing) is mandatory because raw PCM contains
`0x0A` bytes. `PAYLOAD` is UTF-8 single-line JSON for control/result frames and
raw little-endian signed 16-bit mono 16 kHz PCM for `AUDIO`. Frames never
exceed 64 MiB.

**Type tags.**

| Tag | Name | Dir | Payload | Meaning |
|---|---|---|---|---|
| `0x01` | HELLO | daemon→sidecar | JSON | session open + params |
| `0x02` | AUDIO | daemon→sidecar | binary | raw PCM chunk |
| `0x03` | FLUSH | daemon→sidecar | JSON | utterance boundary: finalize + reset, stay connected |
| `0x04` | BYE | daemon→sidecar | JSON | end session, close cleanly (model stays resident) |
| `0x10` | READY | sidecar→daemon | JSON | model loaded + HELLO accepted |
| `0x11` | PARTIAL | sidecar→daemon | JSON | in-progress, replaceable hypothesis |
| `0x12` | FINAL | sidecar→daemon | JSON | committed, punctuated+capitalized utterance text |
| `0x13` | ERROR | sidecar→daemon | JSON | recoverable or fatal (code distinguishes) |

**Session flow.** One streaming toggle = one session:

1. Daemon connects, sends `HELLO` (`{"v":1, "session_id", "sample_rate":16000,
   "encoding":"s16le", "channels":1, "want_partials":bool}`).
2. Sidecar validates against the loaded model, resets streaming state, replies
   `READY`. (The model loads before the socket accepts, so READY is honest.)
3. Daemon streams `AUDIO` frames as mic data arrives (~100 ms chunks; the
   sidecar re-buffers to the model's native 160 ms step).
4. Sidecar emits `PARTIAL` whenever the running hypothesis grows (only if
   `want_partials`). Partials are advisory/replaceable — never committed.
5. On an utterance boundary — daemon `FLUSH`, or the sidecar's own endpointer
   after `eou_silence_ms` of trailing silence — the sidecar finalizes and emits
   `FINAL` with native punctuation + capitalization.
6. Per-utterance state resets, the `utt` counter increments, and the same
   connection continues. Multiple FLUSH/FINAL pairs per session are normal.
7. Daemon sends `BYE`; sidecar closes, frees per-session state, returns to
   `accept()`. **The model stays resident across sessions.**

**Error semantics.** `ERROR{code, message, fatal}`: `busy` / `bad_audio` /
`model_load` are fatal (socket closes; `model_load` exits non-zero so systemd
restarts); `inference` is transient (session continues with reset state). The
daemon treats any socket drop or fatal ERROR as "sidecar unavailable" and falls
back gracefully, keeping the whisper paths usable.

**Robustness rules.** The sidecar prioritizes reading audio and never blocks
indefinitely writing PARTIALs (a dropped partial is superseded by the next).
The daemon sends FLUSH, waits briefly for the matching FINAL, then BYE.
`HELLO`/`READY` carry `"v":1` so future changes are detectable.

### 6. Injection modes for streaming — `[nemotron].inject_mode`

- **`finals`** (default): type each committed FINAL when it arrives — clean,
  phrase-at-a-time output.
- **`live`**: type PARTIALs as you speak. The daemon tracks what's on screen,
  prefix-diffs each revision, and corrects in place (one batched
  backspace+retype per revision burst, coalescing rapid partials). When the
  FINAL arrives it reconciles the tail and commits. Feels like the text keeps
  up with your voice; costs occasional visible corrections when the model
  revises a word retroactively.

### 7. LLM cleanup — optional, off by default

Fully implemented (`llm_postprocess.py`, glossary, per-app context via
`app_context.py`, preset modes via `modes.py`, voice-command-on-selection,
JSONL audit log) but **disabled** by default. It's an OpenAI-compatible
(`/v1/chat/completions`) HTTP client; any failure falls back to the raw
transcript. Point `[llm_postprocess].endpoint` at any local llama.cpp / Ollama /
LM Studio / vLLM server (`toggle_llm.sh` shows the load/unload pattern for a
llama-server systemd unit). Cleanup applies to the press-to-talk path; streaming
stays cleanup-free for latency. ASR with native punctuation means cleanup is a
*polish* choice, not a requirement.

### 8. Text injection — the output seam

[`text_injector.py`](text_injector.py) is the pluggable injection layer:
`wtype` (Wayland), `xdotool` (X11), or `ydotool` (both), selected by
`[wayland].typer` (`auto` picks the first available). The interface is
`type_text()` plus `backspace()`/`edit()` (batched backspace+retype, used by
live mode) and `paste()`. For long text (past
`[wayland].clipboard_paste_threshold`) the daemon switches to a
`wl-copy` + synthetic-paste path with clipboard restore
(`clipboard_paste.py`) — keystroke synthesis gets flaky on very long inputs.

### 9. Config — the control plane

`config_loader.py` builds typed dataclasses from `config.toml`, then deep-merges
`config.local.toml` over it (per-machine: GPU index, endpoints, model). Every
component reads config; nothing hardcodes paths. The sidecar reads its
`[mumble_stt]` table from the *same* file (`mumble_stt/config.py`), so there is
one control plane. Internal non-tunable values live in `constants.py` — no
magic numbers in the logic.

## Install & operate

- **`install.sh`** — interactive, idempotent, conservative. Creates the venv,
  installs Python deps (via `uv`), builds whisper.cpp, **auto-downloads** the
  configured model, installs the systemd **user** unit, prints the keybind
  commands. For missing *system* packages it detects and prints the exact
  command for your distro, then stops — it does not run your package manager.
  The Nemotron sidecar is opt-in (`--with-streaming`) and probed-and-reported,
  not silently built (see the README for the heavy-venv setup).
- **`mumble doctor`** — read-only diagnostic preflight: injector, clipboard
  tools, ffmpeg/ncat, whisper binaries, model presence, backend config,
  services, sockets. Reports PASS/WARN/FAIL with the fix for each.
- **Services** — `mumble.service` (daemon) and, opt-in, `mumble-stt.service`
  (sidecar). Most config changes need a daemon restart; `[mumble_stt]` changes
  need a sidecar restart. Logs via `journalctl --user -u <unit> -f`.

## Extending mumble

- **New ASR backend** → implement the batch or streaming contract (§3),
  register it under a config value, add its settings to config. A heavy engine
  that can't share the light venv should follow the sidecar pattern (§4–5).
- **New text injector** → add it in `text_injector.py` behind the same
  interface (§8).
- **New cleanup model/host** → point `llm_postprocess.endpoint` at it (§7).
- **New client/keybind** → send an IPC verb (§2). That's the whole integration.

## Repo map

```
mumble                 CLI: toggle | stream | status | mode | doctor
whisper_daemon.py      daemon: IPC, recording, backend dispatch, pipeline, injection
backends/              streaming backend clients (nemotron_streaming.py)
mumble_stt/            Nemotron sidecar: server, engine (NeMo), protocol, config
text_injector.py       pluggable typing: wtype/xdotool/ydotool + edit/backspace/paste
config_loader.py       typed config: config.toml + config.local.toml merge
config.toml            default configuration (committed, commented)
config.local.toml      per-machine overrides (gitignored)
constants.py           internal named constants (non-user-tunable)
llm_postprocess.py     optional LLM cleanup client (off by default)
glossary.py modes.py app_context.py clipboard_paste.py   pipeline helpers
stream_dedup.py word_dedup.py   legacy whisper-stream dedup heuristics
calibrate.py benchmark.py benchmarks/   stream-tuning + evaluation tools
waybar_whisper.py      waybar status module
toggle_*.sh            back-compat keybind wrappers over the CLI
mumble.service mumble-stt.service   systemd user units
install.sh build_whisper.sh   setup
tests/                 unit tests
```
