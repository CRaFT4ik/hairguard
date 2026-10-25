"""Console entry point: the full command line, inside the bundle.

Shipped next to the windowed executable because a packaged app that cannot
be asked what went wrong is worse than a script. It is also what the app
spawns for the diagnostic window — a windowed Windows binary is not
guaranteed a usable standard input, and that is the whole channel.
"""
import sys

from hairguard.entry import console_main

if __name__ == "__main__":
    sys.exit(console_main())
