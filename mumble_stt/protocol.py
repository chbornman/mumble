"""mumble-stt wire protocol — the single source of truth for the framing and
message contract between the sidecar (server) and the daemon-side
``nemotron-streaming`` backend (client).

This module is PURE STDLIB (struct, json) on purpose: the daemon imports it
inside mumble's lightweight venv, where torch/NeMo are NOT installed. Nothing
heavy may ever be imported from here. The human-readable spec lives in
the sidecar protocol section of ``ARCHITECTURE.md``; this file is the machine
source of truth.

TRANSPORT
    A SOCK_STREAM unix domain socket. Every frame, in both directions, is:

        byte 0      : TYPE   (uint8)
        bytes 1..4  : LEN    (uint32, big-endian) = length of PAYLOAD
        bytes 5..   : PAYLOAD (LEN bytes)

    PAYLOAD is UTF-8 single-line JSON for control/result frames, and raw
    little-endian signed 16-bit mono 16 kHz PCM for AUDIO frames. Length
    prefixing (not newline framing) is mandatory because PCM contains 0x0A
    bytes.
"""

from __future__ import annotations

import json
import struct
from typing import Any

# Protocol version. Carried on HELLO/READY so future changes are detectable.
PROTOCOL_VERSION = 1

# Sidecar identity string (handy for logs / READY diagnostics).
SIDECAR_NAME = "mumble-stt"

# --- frame type tags --------------------------------------------------------
# daemon -> sidecar
TYPE_HELLO = 0x01   # JSON: session open + params
TYPE_AUDIO = 0x02   # binary: raw PCM chunk
TYPE_FLUSH = 0x03   # JSON: utterance boundary -> finalize + reset, stay connected
TYPE_BYE = 0x04     # JSON: end session, close cleanly
# sidecar -> daemon
TYPE_READY = 0x10   # JSON: model loaded + HELLO accepted
TYPE_PARTIAL = 0x11  # JSON: in-progress hypothesis for current utterance
TYPE_FINAL = 0x12   # JSON: finalized text for an utterance
TYPE_ERROR = 0x13   # JSON: recoverable or fatal error (code distinguishes)

# Human-readable names, for logging.
TYPE_NAMES = {
    TYPE_HELLO: "HELLO",
    TYPE_AUDIO: "AUDIO",
    TYPE_FLUSH: "FLUSH",
    TYPE_BYE: "BYE",
    TYPE_READY: "READY",
    TYPE_PARTIAL: "PARTIAL",
    TYPE_FINAL: "FINAL",
    TYPE_ERROR: "ERROR",
}

# --- error codes ------------------------------------------------------------
ERR_BUSY = "busy"            # another session is active
ERR_BAD_AUDIO = "bad_audio"  # HELLO audio format mismatch
ERR_MODEL_LOAD = "model_load"  # NeMo load failed at startup (fatal, process exits)
ERR_INFERENCE = "inference"  # transient decode failure (non-fatal, state reset)
ERR_INTERNAL = "internal"    # anything else

# 5-byte header: 1-byte type + 4-byte big-endian length.
_HEADER = struct.Struct(">BI")
HEADER_SIZE = _HEADER.size  # 5

# Bound a single frame's payload so a corrupt/hostile length prefix can't make
# us try to allocate gigabytes. 64 MiB is far above any real audio chunk or
# JSON message.
MAX_PAYLOAD = 64 * 1024 * 1024


class ProtocolError(Exception):
    """Raised on a malformed frame (bad length, truncated stream, bad JSON)."""


# --- framing primitives -----------------------------------------------------
def encode_frame(type_tag: int, payload: bytes) -> bytes:
    """Serialize one frame (header + payload) to bytes ready for the socket."""
    if len(payload) > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    return _HEADER.pack(type_tag, len(payload)) + payload


def encode_json_frame(type_tag: int, obj: dict[str, Any]) -> bytes:
    """Serialize a JSON control/result frame. Single-line UTF-8 JSON."""
    payload = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return encode_frame(type_tag, payload)


def decode_json(payload: bytes) -> dict[str, Any]:
    """Parse a JSON frame payload into a dict."""
    try:
        obj = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProtocolError(f"invalid JSON payload: {e}") from e
    if not isinstance(obj, dict):
        raise ProtocolError("JSON payload is not an object")
    return obj


# --- blocking socket readers (used by the stdlib-only daemon client) --------
def _recv_exact(sock, n: int) -> bytes:
    """Read exactly ``n`` bytes from a blocking socket, or raise on EOF."""
    chunks: list[bytes] = []
    remaining = n
    while remaining > 0:
        buf = sock.recv(remaining)
        if not buf:
            raise ProtocolError("connection closed mid-frame")
        chunks.append(buf)
        remaining -= len(buf)
    return b"".join(chunks)


def read_frame(sock) -> tuple[int, bytes]:
    """Read one full frame from a blocking socket. Returns (type_tag, payload).

    Raises ProtocolError on EOF/truncation/oversized length.
    """
    header = _recv_exact(sock, HEADER_SIZE)
    type_tag, length = _HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"frame length {length} exceeds max {MAX_PAYLOAD}")
    payload = _recv_exact(sock, length) if length else b""
    return type_tag, payload


# --- asyncio readers (used by the sidecar server) ---------------------------
async def read_frame_async(reader) -> tuple[int, bytes]:
    """Read one full frame from an asyncio.StreamReader.

    Returns (type_tag, payload). Raises ProtocolError on EOF/oversized length.
    """
    try:
        header = await reader.readexactly(HEADER_SIZE)
    except Exception as e:  # asyncio.IncompleteReadError on clean EOF
        raise ProtocolError("connection closed mid-frame") from e
    type_tag, length = _HEADER.unpack(header)
    if length > MAX_PAYLOAD:
        raise ProtocolError(f"frame length {length} exceeds max {MAX_PAYLOAD}")
    if length:
        try:
            payload = await reader.readexactly(length)
        except Exception as e:
            raise ProtocolError("connection closed mid-frame") from e
    else:
        payload = b""
    return type_tag, payload


# --- message builders (so callers never hand-assemble dicts) ----------------
def hello(session_id: str, sample_rate: int = 16000, encoding: str = "s16le",
          channels: int = 1, want_partials: bool = True) -> bytes:
    return encode_json_frame(TYPE_HELLO, {
        "v": PROTOCOL_VERSION,
        "type": "hello",
        "session_id": session_id,
        "sample_rate": sample_rate,
        "encoding": encoding,
        "channels": channels,
        "want_partials": want_partials,
    })


def flush(utt: int) -> bytes:
    return encode_json_frame(TYPE_FLUSH, {"type": "flush", "utt": utt})


def bye(session_id: str) -> bytes:
    return encode_json_frame(TYPE_BYE, {"type": "bye", "session_id": session_id})


def ready(model: str, sample_rate: int = 16000, att_context: str = "default") -> bytes:
    return encode_json_frame(TYPE_READY, {
        "v": PROTOCOL_VERSION,
        "type": "ready",
        "model": model,
        "sample_rate": sample_rate,
        "att_context": att_context,
    })


def partial(utt: int, text: str, stable_prefix_len: int = 0) -> bytes:
    return encode_json_frame(TYPE_PARTIAL, {
        "type": "partial",
        "utt": utt,
        "text": text,
        "stable_prefix_len": stable_prefix_len,
    })


def final(utt: int, text: str) -> bytes:
    return encode_json_frame(TYPE_FINAL, {"type": "final", "utt": utt, "text": text})


def error(code: str, message: str, fatal: bool = False) -> bytes:
    return encode_json_frame(TYPE_ERROR, {
        "type": "error",
        "code": code,
        "message": message,
        "fatal": fatal,
    })


def audio(pcm: bytes) -> bytes:
    """Frame a raw PCM chunk (s16le mono 16 kHz) as an AUDIO frame."""
    return encode_frame(TYPE_AUDIO, pcm)
