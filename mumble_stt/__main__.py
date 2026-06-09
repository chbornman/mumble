"""Allow ``python -m mumble_stt`` as a convenience alias for the server."""

from mumble_stt.server import main

if __name__ == "__main__":
    raise SystemExit(main())
