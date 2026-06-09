"""mumble-stt — the Nemotron streaming ASR sidecar for mumble.

A standalone long-lived process that holds the cache-aware FastConformer-RNNT
model resident and emits PARTIAL + FINAL transcript segments over a local unix
socket as audio streams in. It runs in mumble's HEAVY venv (NeMo + torch); the
daemon talks to it through ``protocol.py`` (pure stdlib) so the heavy deps stay
isolated behind the socket.

Entrypoint: ``python -m mumble_stt`` (or ``mumble_stt/server.py``).
Wire contract: ``mumble_stt.protocol`` (machine) / the protocol section of ``ARCHITECTURE.md``
(human).
"""

from . import protocol  # noqa: F401  (re-export the dependency-free wire module)

__all__ = ["protocol"]
__version__ = "0.1.0"
