"""hairguard.quiet — keep native library chatter out of the console.

MediaPipe, TensorFlow Lite and absl announce themselves on every start:
XNNPACK delegates, feedback tensors, ROI projection notes. A dozen lines
nobody can act on, printed before anything useful, which is enough to hide
the messages that do matter — which camera was chosen, whether collection
is running, why a clip failed.

None of it can be stopped from Python. These libraries write to the
operating system's stderr from C++, below ``sys.stderr``, so logging
configuration and output redirection at the Python level both miss it.
Silencing the file descriptor itself is the only thing that works.

Python exceptions are unaffected: a traceback is printed by the interpreter
after the block exits, once the descriptor is back.
"""
from __future__ import annotations

import contextlib
import os
import sys
from typing import Iterator

VERBOSE_ENV = "HAIRGUARD_VERBOSE"


@contextlib.contextmanager
def suppressed_native_stderr() -> Iterator[None]:
    """Silence writes to the OS-level stderr for the duration of the block.

    Set ``HAIRGUARD_VERBOSE=1`` to keep the native logs while debugging.
    """
    if os.environ.get(VERBOSE_ENV):
        yield
        return

    try:
        sys.stderr.flush()
    except Exception:
        pass

    try:
        saved = os.dup(2)
    except OSError:
        # No real stderr to silence (pythonw, a detached service). Nothing
        # is being printed anyway.
        yield
        return

    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 2)
        yield
    finally:
        try:
            sys.stderr.flush()
        except Exception:
            pass
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)
