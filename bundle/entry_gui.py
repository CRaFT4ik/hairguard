"""Windowed entry point: the icon a person double-clicks.

A separate file from the console one so that the two executables differ in
exactly one thing — whether a bare launch means "start" and whether there is
a console to print to. See ``hairguard.entry``.
"""
import sys

from hairguard.entry import gui_main

if __name__ == "__main__":
    sys.exit(gui_main())
