# Contributing to mumble

Thanks for your interest. mumble is small and modular on purpose — most useful
contributions are a new backend or a new client wired through an existing seam,
not surgery on the daemon. This doc covers how to get set up, run the tests, and
the conventions to match.

If you're touching the internals, read [ARCHITECTURE.md](ARCHITECTURE.md) first.
It is the map of the components, the IPC wire format, and the extension points
referenced below.

## Getting set up

mumble uses a `uv`-managed virtualenv. The installer creates it for you:

```bash
git clone https://github.com/chbornman/mumble.git
cd mumble
./install.sh --dry-run   # see what it would do, no changes
./install.sh             # real run: builds the venv, whisper.cpp, the service
```

If you only want the Python environment (e.g. to run tests without building
whisper.cpp or touching systemd):

```bash
uv venv .venv
uv pip install -r requirements.txt
```

Per-machine settings (GPU index, model, endpoints) belong in
`config.local.toml`, which is gitignored and deep-merged over `config.toml`.
Never commit machine-specific paths into `config.toml`.

## Running the tests

The unit tests run against the project venv with the standard library's test
runner:

```bash
.venv/bin/python -m unittest discover -s tests
```

Run a single module while iterating:

```bash
.venv/bin/python -m unittest tests.test_word_dedup
```

The pure-protocol path of the Nemotron sidecar can be exercised without a GPU or
the heavy NeMo venv:

```bash
python mumble_stt/server.py --selftest
```

Changes to the NeMo engine itself must also be exercised in the heavy venv on
the device paths they claim to support. In particular, do not infer CPU support
from CUDA success (or the reverse). Record the model state (cold or already
loaded), audio duration, processing duration, post-audio tail for paced tests,
resident DRAM/VRAM, and hardware when publishing performance results.

Add tests for new behavior in `tests/`, following the existing
`test_<module>.py` layout and `unittest.TestCase` style.

Vocabulary changes need separate coverage for the bounded batch Whisper
prompt, the optional structured LLM glossary, and the opt-in Nemotron context
bias path. They are distinct consumers: Whisper includes mapping sources and
destinations, while Nemotron includes only destinations, and quoted cleanup
rules are LLM-only. Keep daemon and benchmark prompt construction identical.

Nemotron vocabulary tests must cover the disabled path, CUDA/Triton and
CPU/PyTorch decoder configuration, duplicate handling, mapping selection,
phrase-count refusal, desired technical terms, and ordinary-speech negative
cases for over-biasing. A successful load or unchanged generic transcript is a
smoke test, not evidence of better accuracy; report bias-off and bias-on results
without implying improvement unless a representative term set demonstrates it.

## The extension seams

These are the supported ways to extend mumble. Each is a small, documented
interface — you should not need to modify `whisper_daemon.py` core logic.

### Add an ASR backend

A backend turns audio into text; the daemon doesn't care how.

- **Batch backend** (press-to-talk): "given a WAV, return text." Model after
  `whisper-cli` / `whisper-server`.
- **Streaming backend** (live toggle): "given a mic stream, emit segments as you
  go." Model after `whisper-stream` / the `nemotron-streaming` client in
  [`backends/`](backends/).

Register it under a value of `[backend].type` (batch) or
`[backend].streaming_backend` (streaming) in `config.toml`, and add any settings
it needs to the config and to `config_loader.py`'s dataclasses. See
ARCHITECTURE "ASR backends" and "Extending mumble."

A heavy engine that can't share the light venv (like Nemotron's NeMo/PyTorch
stack) should follow the **sidecar** pattern: its own long-lived process and
venv, talking to the daemon over a socket. See the sidecar sections of
[ARCHITECTURE.md](ARCHITECTURE.md) for the reference implementation
(`mumble_stt/`), and [`mumble_stt/requirements.txt`](mumble_stt/requirements.txt)
for building the heavy venv if you want to test the streaming path:

```bash
python -m venv .venv-stt
.venv-stt/bin/pip install -r mumble_stt/requirements.txt
```

### Add a text injector

mumble needs exactly one working injector. Injection is pluggable behind
[`text_injector.py`](text_injector.py) (`type_text` / `backspace` / `edit` /
`paste`) and selected by `[wayland].typer`. Add the new tool there and document
its config value. Keep `wtype` / `xdotool` / `ydotool` working.

### Add a keybind / client

Clients are tiny: send a text verb to the daemon's unix socket (e.g. `TOGGLE`,
`STREAM`, `STATUS`). The `toggle_*.sh` scripts are the reference. That's the
entire integration — no new daemon code needed.

### Add a config setting

Add it in three places: the dataclass in `config_loader.py`, the key (commented)
in `config.toml`, and the loader that wires them together. Everything else reads
config, so the value then propagates automatically.

## The no-magic-install philosophy

This is a hard design rule, not a preference — please preserve it in any change
to `install.sh` or the docs:

- The **README is the source of truth** for what to install.
- The installer is **interactive, idempotent, and conservative**. It may build
  the venv, install Python deps via `uv`, build whisper.cpp, download models, and
  install the systemd **user** unit.
- It must **never** run the user's package manager, never `sudo`, and never edit
  the user's compositor config. For missing **system** packages it **detects and
  prints** the exact command for the user's distro, then stops.
- Heavy, optional components (the Nemotron sidecar and its multi-GB venv/model)
  stay **opt-in** and are probed-and-reported, not silently built.

If a change would make the installer touch system state on the user's behalf,
it's the wrong change.

## Code style

- Match the surrounding code. Python here is plain, dependency-light, and
  config-driven — "no magic numbers; everything comes from `config.toml`."
- Keep new dependencies minimal; the daemon venv is intentionally lightweight.
- Prefer extending an existing seam over adding a new abstraction.
- Keep commits focused and explain the *why* in the message.

## Reporting issues

Open an issue at https://github.com/chbornman/mumble/issues. For dictation or
streaming problems, include your distro, compositor, `config.toml` (minus
secrets), the chosen backends, and relevant lines from
`journalctl --user -u mumble.service`.

## License

By contributing, you agree your contributions are licensed under the project's
[MIT License](LICENSE).
