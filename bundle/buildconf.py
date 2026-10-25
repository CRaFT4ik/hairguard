"""bundle.buildconf — what a HairGuard bundle contains, and what it must not.

Kept apart from ``hairguard.spec`` so it can be tested without running a
build, because the two things it decides are both expensive to get wrong.

**torch must not ship.** It is 2.8 GB. The repository contains the training
code beside the app, and ``hairlab.model`` imports torch at module scope, so
a bundler following imports from ``hairlab`` would take it. The app needs
none of it: inference goes through onnxruntime, which is 40 MB, and the ONNX
graph carries its own input normalisation. The exclusions below are
therefore not an optimisation, they are the difference between a shippable
artefact and a 3 GB download for a tray utility. Because an exclude list is
a claim rather than a fact, :func:`verify_bundle` checks the finished tree.

**The trained model must ship.** Without it the app falls back to the
geometric rule — 53.9% of episodes caught against 86.7% — and does so while
starting normally and looking fine. The developer's machine has the
artefacts, so the failure would never show up locally; a user would simply
have a worse detector and no way to know. :func:`preflight` refuses to build
without them. That is a deliberate choice of a failed build over a
successful one that ships something quietly broken.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

APP_NAME = "HairGuard"
"""The windowed executable: the icon a person double-clicks."""

CONSOLE_NAME = "hairguard-cli"
"""The console executable: the whole command line, for diagnosis.

Shipped alongside rather than instead, because a packaged app that cannot be
asked what went wrong is worse than a script. It is also what the app spawns
for the diagnostic window, a windowed Windows binary having no guaranteed
standard input.

Not plain ``hairguard``, which would be the obvious name and was the first
attempt: Windows filenames are case-insensitive, so ``hairguard.exe`` and
``HairGuard.exe`` are the same file and the second one built silently
replaced the first. The suffix is the price of having both in one folder.
"""

BUNDLE_IDENTIFIER = "com.eldar.hairguard"

#: Files that make up the trained detector. The ``.onnx`` is the graph and
#: the ``.onnx.data`` its external weights — 4.7 KB and 395 KB, and the graph
#: alone would load, fail its own trial inference and fall back to the rule.
TRAINED_MODEL_FILES = ("model.json", "twistnet.onnx", "twistnet.onnx.data")

#: Deliberately not shipped: 400 KB of pickled torch tensors that only
#: ``hairlab`` can read, and only in order to retrain.
UNSHIPPED_MODEL_FILES = ("twistnet.ckpt",)

#: The MediaPipe landmark models, 21 MB together. Downloadable at runtime,
#: bundled anyway: the downloads happen before the tray icon exists, so on a
#: fresh machine the user's first impression is an app doing nothing for
#: twenty seconds, and on a plane it is an app that does not work.
MEDIAPIPE_TASKS = (
    "hand_landmarker.task",
    "face_landmarker.task",
    "pose_landmarker_full.task",
)

#: Imported through ``find_spec`` (``hairguard.labpath``), behind
#: platform checks (``pystray``, ``plyer``), or only ever named as a string
#: — all cases static analysis cannot be relied on for.
HIDDEN_IMPORTS = [
    "onnxruntime",
    # The feature extractor the model was trained with. twistnet refuses a
    # model whose feature width disagrees with it, so without this the
    # bundle starts on the rule.
    "hairlab.features",
    # Dataset collection, which is in the shipped tray menu: it is how a user
    # trains a detector on their own hands.
    "hairlab.capture",
    "hairlab.clipstore",
    "hairlab.collector",
    "hairlab.state",
    "hairlab.writer",
    "hairguard.debugviewer",
]

#: Added per platform, where the backend is chosen at runtime.
PLATFORM_HIDDEN_IMPORTS = {
    "win32": [
        "pystray._win32",
        "plyer.platforms.win.notification",
        "winsound",
    ],
    "darwin": [
        "pystray._darwin",
        "plyer.platforms.macosx.notification",
    ],
    "linux": [
        "pystray._xorg",
        "pystray._appindicator",
        "pystray._gtk",
        "plyer.platforms.linux.notification",
    ],
}

EXCLUDES = [
    # --- the training stack, which is the whole point of this list ---
    "torch",
    "torchvision",
    "torchaudio",
    "torchmetrics",
    "pytorch_lightning",
    "lightning",
    "lightning_fabric",
    "lightning_utilities",
    "sklearn",
    "scipy",
    # The exporter, not the runtime. onnxruntime is required and is a hidden
    # import above; these two are only needed to *produce* the graph.
    "onnx",
    "onnxscript",
    "tensorboard",
    "tensorboardX",
    # --- the modules that would drag the above in ---
    "hairlab.model",
    "hairlab.train",
    "hairlab.dataset",
    "hairlab.evaluation",
    "hairlab.truncation",
    "hairlab.server",
    # --- the labelling web UI, a developer tool ---
    "fastapi",
    "uvicorn",
    "starlette",
    "pydantic",
    "pydantic_core",
    # --- things that arrive by accident and are large ---
    "pandas",
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
    "setuptools",
    "pip",
    # Matplotlib's test suite, 5 MB of reference images. The library itself
    # has to stay — see MATPLOTLIB_IS_REQUIRED below.
    "matplotlib.tests",
]

MATPLOTLIB_IS_REQUIRED = True
"""Why 29 MB of plotting library is in a webcam tray utility.

``mediapipe.tasks.python.vision.__init__`` imports ``drawing_styles``, which
imports ``drawing_utils``, which does ``import matplotlib.pyplot`` at module
scope. Nothing in HairGuard draws a plot and nothing calls those functions,
but the app imports ``PoseLandmarker`` from that package and Python runs the
package's ``__init__`` to get there. There is no import path around it.

It was excluded in the first build and the result failed on its first
import with ``ModuleNotFoundError: No module named 'matplotlib'`` — from
inside mediapipe, four frames below anything of ours. Recorded here because
it looks exactly like dead weight somebody should delete.
"""

#: Checked against the finished tree. Each entry is a path component that
#: must not appear. Matched whole, so "onnx" does not condemn "onnxruntime".
FORBIDDEN_COMPONENTS = (
    "torch",
    "torchvision",
    "torchaudio",
    "torchmetrics",
    "pytorch_lightning",
    "lightning",
    "lightning_fabric",
    "sklearn",
    "scipy",
    "onnx",
    "onnxscript",
    "pandas",
)


#: MediaPipe's entire implementation, by platform. This build of mediapipe
#: is a ctypes wrapper around one 28 MB shared library loaded through
#: ``importlib.resources.files('mediapipe.tasks.c')`` — a path computed at
#: runtime, which no import-following bundler can see. Left out, the bundle
#: imports mediapipe successfully and then fails on the first detector.
MEDIAPIPE_LIBRARIES = (
    "libmediapipe.dll",
    "libmediapipe.dylib",
    "libmediapipe.so",
)

MEDIAPIPE_LIBRARY_TARGET = "mediapipe/tasks/c"


def _missing(directory: Path, names: Iterable[str]) -> list[str]:
    return [name for name in names if not (directory / name).is_file()]


def mediapipe_native_libraries() -> list[tuple[str, str]]:
    """``(source, target)`` for MediaPipe's shared library, placed where it looks."""
    try:
        import mediapipe
    except ImportError:
        return []
    base = Path(mediapipe.__file__).parent / "tasks" / "c"
    return [
        (str(base / name), MEDIAPIPE_LIBRARY_TARGET)
        for name in MEDIAPIPE_LIBRARIES
        if (base / name).is_file()
    ]


# ---------------------------------------------------------------------------
# Before the build
# ---------------------------------------------------------------------------


def preflight(
    repo_root: Path,
    models_dir: Path,
    *,
    require_landmark_models: bool = True,
) -> list[str]:
    """Everything wrong with the sources, as sentences. Empty means go.

    Returns all the problems rather than the first, because a build takes
    minutes and finding one fault per attempt would be unkind.
    """
    repo_root = Path(repo_root)
    models_dir = Path(models_dir)
    problems: list[str] = []

    trained = repo_root / "hairguard" / "models"
    missing = _missing(trained, TRAINED_MODEL_FILES)
    if missing:
        problems.append(
            f"В {trained} нет файлов обученной модели: {', '.join(missing)}. "
            "Собранное приложение запустилось бы на пороговом правиле — "
            "53.9% эпизодов против 86.7% у модели — и выглядело бы при этом "
            "работающим. Обучить: python -m hairlab train"
        )

    features = repo_root / "hairlab" / "features.py"
    if not features.is_file():
        problems.append(
            f"Нет {features} — извлекателя признаков, с которым обучалась "
            "модель. twistnet сверяет с ним ширину вектора и без него "
            "откажется от модели, снова оставив правило."
        )

    if require_landmark_models:
        missing_tasks = _missing(models_dir, MEDIAPIPE_TASKS)
        if missing_tasks:
            problems.append(
                f"В {models_dir} нет моделей MediaPipe: "
                f"{', '.join(missing_tasks)}. Их можно не включать "
                "(--no-landmark-models), но тогда при первом запуске "
                "приложение 20 секунд молча их скачивает, а без сети не "
                "работает вовсе. Запустите приложение один раз, чтобы они "
                "появились."
            )

    try:
        import onnxruntime  # noqa: F401
    except ImportError as exc:
        problems.append(
            f"onnxruntime недоступен ({exc}) — без него собранное приложение "
            "не сможет запустить модель и останется на правиле."
        )

    if not mediapipe_native_libraries():
        problems.append(
            "Не найдена нативная библиотека MediaPipe "
            f"({' / '.join(MEDIAPIPE_LIBRARIES)}) рядом с пакетом mediapipe. "
            "Она загружается по вычисленному пути, и без неё сборка "
            "импортирует mediapipe нормально, а падает на первом детекторе."
        )

    return problems


def datas(
    repo_root: Path,
    models_dir: Path,
) -> list[tuple[str, str]]:
    """``(source, target_directory)`` pairs for the model artefacts.

    The targets mirror where the code looks: ``twistnet.BUNDLED_DIR`` is
    ``Path(__file__).parent / "models"`` and ``assets.bundled_dir()`` is
    ``Path(__file__).parent / "assets"``, both of which resolve inside the
    bundle because the bundler rewrites the package's path.
    """
    repo_root = Path(repo_root)
    models_dir = Path(models_dir)
    entries: list[tuple[str, str]] = []

    trained = repo_root / "hairguard" / "models"
    for name in TRAINED_MODEL_FILES:
        source = trained / name
        if source.is_file():
            entries.append((str(source), "hairguard/models"))

    for name in MEDIAPIPE_TASKS:
        source = models_dir / name
        if source.is_file():
            entries.append((str(source), "hairguard/assets"))

    return entries


# ---------------------------------------------------------------------------
# After the build
# ---------------------------------------------------------------------------


def _internal(dist_dir: Path) -> Path:
    """Where PyInstaller 6 puts the libraries, or the root for older layouts."""
    candidate = dist_dir / "_internal"
    return candidate if candidate.is_dir() else dist_dir


def _macos_root(dist_dir: Path) -> Path:
    """The directory holding the executables, inside a .app or not."""
    inside = dist_dir / "Contents" / "MacOS"
    return inside if inside.is_dir() else dist_dir


def verify_bundle(dist_dir: Path) -> list[str]:
    """Everything wrong with the finished tree, as sentences. Empty means good.

    The exclude list is a claim; this is the fact. It is also the answer to
    "prove torch is not in there", which is not a question a configuration
    file can settle.
    """
    dist_dir = Path(dist_dir)
    problems: list[str] = []

    if not dist_dir.is_dir():
        return [f"Каталог сборки {dist_dir} не найден."]

    # --- nothing from the training stack ---
    offenders: dict[str, int] = {}
    for root, dirs, files in os.walk(dist_dir):
        relative = Path(root).relative_to(dist_dir)
        parts = {part.lower() for part in relative.parts}
        hit = parts.intersection(FORBIDDEN_COMPONENTS)
        if not hit:
            continue
        name = sorted(hit)[0]
        size = sum(
            (Path(root) / file).stat().st_size
            for file in files
            if (Path(root) / file).is_file()
        )
        offenders[name] = offenders.get(name, 0) + size
    for name, size in sorted(offenders.items()):
        problems.append(
            f"В сборке найден {name} ({size / 1e6:.1f} МБ) — он нужен только "
            "для обучения и не должен попадать в приложение."
        )

    # --- the trained detector, without which the bundle is quietly worse ---
    internal = _internal(dist_dir)
    trained = internal / "hairguard" / "models"
    missing = _missing(trained, TRAINED_MODEL_FILES)
    if missing:
        problems.append(
            f"В сборке нет файлов модели: {', '.join(missing)}. Приложение "
            "запустится на правиле и не скажет пользователю ничего, кроме "
            "одной строки в журнале."
        )

    # Note on what is *not* checked here: hairlab.features and the rest of
    # the pure Python goes into the bundle's archive, not onto disk, so no
    # amount of walking this tree can establish that it arrived. That is
    # what `hairguard-cli doctor` is for, and the build runs it against the
    # finished artefact with a home directory pointed somewhere empty.

    # --- both executables ---
    exe_root = _macos_root(dist_dir)
    present = {
        path.stem: path
        for path in exe_root.iterdir()
        if path.is_file() and path.suffix.lower() in ("", ".exe")
    }
    for name in (APP_NAME, CONSOLE_NAME):
        if name not in present:
            problems.append(
                f"В сборке нет исполняемого файла {name} (есть: "
                f"{', '.join(sorted(present)) or 'ничего'}). "
                + (
                    "Без него упакованное приложение нельзя ни о чём "
                    "спросить — ни про модель, ни про журнал — и окно "
                    "диагностики не открыть."
                    if name == CONSOLE_NAME
                    else "Это то, что пользователь запускает двойным щелчком."
                )
            )
    # Two names that differ only in case are one file on Windows, which is
    # how the first build quietly shipped a single executable.
    if APP_NAME.lower() == CONSOLE_NAME.lower():
        problems.append(
            f"{APP_NAME} и {CONSOLE_NAME} отличаются только регистром — "
            "в Windows это один и тот же файл."
        )

    return problems


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def total_size(path: Path) -> int:
    """Bytes on disk under ``path``."""
    total = 0
    for root, _dirs, files in os.walk(path):
        for file in files:
            candidate = Path(root) / file
            try:
                total += candidate.stat().st_size
            except OSError:
                pass
    return total


def largest_components(path: Path, limit: Optional[int] = None) -> list[tuple[str, int]]:
    """``(name, bytes)`` for each immediate child, biggest first.

    "How big is it" is the deliverable; "what dominates it" is the only
    useful follow-up, and the answer is usually one or two directories.
    """
    path = Path(path)
    root = _internal(path)
    components: list[tuple[str, int]] = []
    for child in root.iterdir():
        if child.is_dir():
            components.append((child.name, total_size(child)))
        else:
            try:
                components.append((child.name, child.stat().st_size))
            except OSError:
                pass
    components.sort(key=lambda item: item[1], reverse=True)
    return components[:limit] if limit else components
