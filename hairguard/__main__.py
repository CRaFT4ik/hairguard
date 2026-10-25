"""``python -m hairguard`` — the command line, with a copy of its output
in ``~/.hairguard/hairguard.log``.

The log copy is not only for the packaged build. The app also reaches this
entry point when it spawns its own diagnostic window, and a console that has
scrolled away answers no questions about yesterday.
"""
import sys

from hairguard.entry import console_main

sys.exit(console_main())
