"""hairguard.labpath — make ``hairlab`` importable when it is sitting right there.

The two packages live side by side in one repository, but the editable
install on this machine maps only ``hairguard`` (its ``top_level.txt`` lists
one name). ``hairlab`` is therefore importable only when the current working
directory happens to be the repository root, which is true of
``python -m hairguard start`` typed in the project and of the test suite, and
false of a console script launched from anywhere else — including autostart,
which uses ``cwd=None`` whenever an installed ``hairguard`` script exists.

That matters much more now than it did. ``hairlab.features`` is the feature
extractor the model was trained with, so without it the app loads no model
and falls back to the geometric rule: 54% of episodes caught instead of 87%,
decided by which directory the app was started from. The user would see a
detector that works well when they launch it by hand and poorly at login,
with nothing to connect the two.

The real fix is for the install to carry both packages. Until it does, this
closes the trapdoor: ``hairlab`` is looked for in exactly one place — beside
the ``hairguard`` package itself — so nothing is guessed and no unrelated
directory can be dragged onto ``sys.path``.

Copying ``hairlab.features`` into ``hairguard`` would also have made the
import work, and would have been the worst possible answer: two copies of
the feature extractor is precisely the train/serve skew the model cannot
survive, and it would drift silently.
"""
from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Optional

_PACKAGE = "hairlab"


def _importable() -> bool:
    try:
        return importlib.util.find_spec(_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def ensure_hairlab_importable() -> Optional[str]:
    """Put the repository root on ``sys.path`` if that is what is missing.

    Returns a note describing what was done, or None when nothing needed
    doing — which is the normal case for a correctly installed copy.
    """
    if _importable():
        return None

    # <root>/hairguard/labpath.py -> <root>
    root = Path(__file__).resolve().parent.parent
    if not (root / _PACKAGE / "__init__.py").is_file():
        return None

    entry = str(root)
    if entry not in sys.path:
        # Appended, not prepended: this is a repair, and it must not get the
        # chance to shadow a properly installed package of the same name.
        sys.path.append(entry)
    importlib.invalidate_caches()

    if not _importable():
        return None
    return f"{_PACKAGE} найден рядом с hairguard ({root}) и добавлен в sys.path"
