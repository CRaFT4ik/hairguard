"""hairguard.assets — finding the MediaPipe landmark models.

Three ``.task`` files, 21 MB together, one per landmarker. They used to be
downloaded from Google's CDN into ``~/.hairguard/models`` the first time each
detector was constructed, with three near-identical copies of the same
download-if-missing block.

That is fine on a developer's machine. It is not fine as the first thing a
packaged app does: the downloads happen before the tray icon exists, so what
the user sees while they run is nothing at all, and on a plane or behind a
proxy they do not run. So a build copies the three files into the bundle and
the order below makes a cached copy, a bundled copy and a download all work
without the app needing to know which situation it is in.

The user's cache is preferred over the bundle on purpose. A file in
``~/.hairguard/models`` is either one this app downloaded or one the user put
there deliberately; the bundled copy is pinned to whenever the build was
made and can only be the older of the two.
"""
from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Optional


class AssetUnavailable(RuntimeError):
    """A landmark model is neither present nor obtainable."""


def user_dir() -> Path:
    """Where downloads are cached, beside the config and the database."""
    return Path.home() / ".hairguard" / "models"


def bundled_dir() -> Path:
    """Where a build puts its copies.

    Inside the package, which is what makes it resolve in a bundle: the
    bundler rewrites ``__file__`` to point into the extracted tree, and the
    assets are placed there relative to the package. A source checkout has no
    such directory and falls through to the cache and the download.
    """
    return Path(__file__).parent / "assets"


def ensure_task_file(
    name: str,
    url: str,
    *,
    download: Optional[Callable[[str, str], None]] = None,
) -> str:
    """Return a local path to ``name``, fetching it only if it is nowhere.

    Raises :class:`AssetUnavailable` with the file, the address and the
    underlying error, because this runs before there is a tray icon or a
    window and the line it writes to the log is the only explanation the user
    will get for nothing having appeared.
    """
    cached = user_dir() / name
    if cached.is_file():
        return str(cached)

    bundled = bundled_dir() / name
    if bundled.is_file():
        return str(bundled)

    fetch = download if download is not None else urllib.request.urlretrieve
    cached.parent.mkdir(parents=True, exist_ok=True)
    print(f"[assets] Загружаю {name} ({url})…")
    try:
        fetch(url, str(cached))
    except Exception as exc:
        # A partial file would be found by the next run, used, and fail
        # somewhere far less obvious than here.
        try:
            cached.unlink(missing_ok=True)
        except OSError:
            pass
        raise AssetUnavailable(
            f"не удалось получить {name}: {exc}. "
            f"Файл нужен для распознавания; скачайте его вручную из {url} "
            f"и положите в {user_dir()}"
        ) from exc
    print(f"[assets] {name} сохранён в {cached}")
    return str(cached)
