# TODO

Open roadmap items. Everything from the original v1 roadmap (LLM
post-processing, per-app context, preset modes, calibration, clipboard
threshold, voice command mode, and legacy stream dedup) has shipped. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the current design.

## Completed in the current vocabulary pass

- **Whisper prompt hygiene.** The daemon and benchmark now share the glossary
  parser, case-insensitive deduplication, whole-term suffix prioritization, and
  an explicit 896-character prompt budget. Later personal/project entries take
  priority over older generic entries.
- **Opt-in Nemotron context biasing.** The sidecar can build a static global
  RNNT boosting tree from vocabulary literals and mapping destinations. CPU
  mode uses the non-Triton PyTorch implementation. A live CPU smoke test with
  a 371-phrase snapshot succeeded; the expanded file now resolves to 407.
- **Baseline A/B measurement.** The same 11-second sample produced identical
  text with biasing off and on. Off took 11.322 seconds end to end with a 1.024
  second paced tail; on took 11.923 seconds with a 1.476 second paced tail.
  This validates operation and quantifies overhead, not recognition quality.

## Current priority

### Validate and tune Nemotron vocabulary biasing

Keep the feature disabled by default until representative speech shows a real
technical-term accuracy gain without excessive ordinary-word substitutions.
The next useful evidence is term recall plus negative/over-bias cases across
several bias strengths. The speaker-specific recording corpus remains parked
below because it requires user participation.

Per-project or per-application vocabulary profiles are also deferred until the
single global vocabulary has measurable benefits that justify the extra
selection machinery.

### Internal naming cleanup (`whisper_*` → `mumble_*`)

The daemon module (`whisper_daemon.py`), IPC socket
(`/tmp/whisper_daemon.sock`), flag files (`/tmp/whisper_recording` and
`/tmp/whisper_streaming`), log file, and logger channel predate the rename.
Changing these breaks deployed keybinds, Waybar modules, and flag watchers, so
do it as one migration with compatibility aliases or a clearly documented
cutover. Do not mix this mechanical migration with recognition changes.

### Hosted LLM cleanup recipe

`[llm_postprocess]` is fully wired but disabled by default. A resident cleanup
model competes with ASR for memory, so document a recommended remote-host or
on-demand `toggle_llm.sh` recipe only after one is deployed and measured. This
Wispr-like cleanup is optional post-recognition polish; it is not required for
either ASR path and must remain separate from low-latency Nemotron streaming.

## Parked / requires a decision or user input

- **Voice-recorded technical-term regression corpus.** This is the best way to
  measure vocabulary improvements for the actual speaker, but recording the
  reference phrases requires user participation. Keep parked until requested.
- **Deterministic correction of streaming finals.** Mapping known mishearings
  after recognition is a CPU-safe fallback, but first validate the implemented
  native context biasing. Add this only if that path proves insufficient; avoid
  maintaining two correction systems preemptively.
- **Streaming + LLM post-processing.** Cleanup behind live streaming adds
  multi-second latency and conflicts with the current type-as-you-speak UX.
  Revisit only with a concrete preview-before-commit design. Live partial
  corrections are ASR revisions, not evidence that an LLM is active.
- **LLM-generated activity summary.** A second LLM call per turn is not
  justified while per-app context covers most of the benefit.
- **Keyring-based API-key storage.** Relevant only if a cloud LLM endpoint is
  supported.
- **Continuous-mode API safety guards.** Relevant only if continuous dictation
  and cloud endpoints are both supported.
- **Per-language Whisper prompts.** The current deployment is English-only.
