"""Always-on SeaHunter edge service command-line defaults."""

from __future__ import annotations

import sys

from .video_replay import main as replay_main


def main(argv: list[str] | None = None) -> int:
    """Run real-time replay with preview and unlimited reconnects by default."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    defaults = ["--realtime", "--preview", "--max-reconnect-attempts", "-1"]
    return replay_main([*defaults, *arguments])


if __name__ == "__main__":
    raise SystemExit(main())
