"""Daemon-side ASR backend clients.

Backends in this package run inside mumble's LIGHT venv (the daemon process).
The streaming backend here is a thin unix-socket client to the mumble-stt
sidecar and imports ONLY ``mumble_stt.protocol`` (pure stdlib) — no torch/NeMo
ever crosses into the daemon.
"""
