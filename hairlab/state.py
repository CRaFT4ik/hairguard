"""hairlab.state — the shared on/off switch for dataset collection.

Collection used to be reachable only through the tray menu, with no way to
check afterwards whether it had actually started. When clips failed to
appear there was no way to tell a missed click from a broken switch.

So the answer lives in a file. The running app applies it, the command line
writes it, and anyone can read it. The file is the truth; the tray and the
CLI are both just ways of setting it.

Off is the answer to anything unexpected — an absent file, a mangled one, an
unreadable directory. A recording switch must never turn itself on.
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_PATH = Path.home() / ".hairguard" / "collect.state"

_ON = "on"
_OFF = "off"


class CollectionState:
    """The on/off switch, shared between processes through a small file."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self.path = Path(path)

    def read(self) -> bool:
        """True only when the file plainly says so."""
        try:
            return self.path.read_text(encoding="utf-8").strip().lower() == _ON
        except (OSError, ValueError):
            return False

    def write(self, enabled: bool) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(_ON if enabled else _OFF, encoding="utf-8")
