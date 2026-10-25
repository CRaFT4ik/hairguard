"""Build a double-clickable HairGuard, then prove it is the right one.

    python bundle/build.py

Three things happen that a bare ``pyinstaller`` run would not do, each
because of a specific way this can go wrong.

*The sources are checked first.* Without the trained model the app falls
back to the geometric rule and catches 53.9% of episodes instead of 86.7%,
while starting normally and looking fine. That failure is invisible on the
machine the build happens on, because the artefacts are there. So a build
without them is refused rather than shipped.

*The result is checked afterwards.* An exclude list is a claim. torch is
2.8 GB and ``hairlab.model`` imports it at module scope, one package away
from the feature extractor the app does need, so "torch is not in the
bundle" is a fact about a directory tree and is established by walking it.

*The packaged app is asked what detector it will use.* With a home directory
pointed at an empty temporary folder, so it cannot find the developer's own
model and must answer from what was bundled. This is the only check that
actually catches a bundle that would run degraded on someone else's machine.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bundle import buildconf  # noqa: E402

SPEC = Path(__file__).resolve().parent / "hairguard.spec"
WORK_DIR = REPO_ROOT / "build"
DIST_DIR = REPO_ROOT / "dist"


# Windows consoles still default to a single-byte code page, and every
# message in this script is Russian. Asked for UTF-8 once, here, rather than
# discovered as a crash two minutes into a build.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def say(message: str = "") -> None:
    print(message, flush=True)


def rule(title: str) -> None:
    say()
    say(f"-- {title} " + "-" * max(0, 66 - len(title)))


def human(size: int) -> str:
    return f"{size / 1e6:.1f} МБ" if size < 1e9 else f"{size / 1e9:.2f} ГБ"


# ---------------------------------------------------------------------------
# Icon
# ---------------------------------------------------------------------------


def make_icon(target: Path) -> Path | None:
    """Render the tray icon to a file the bundler can embed.

    The same green circle the tray draws, so the thing in the Start menu and
    the thing beside the clock are recognisably one application. Windows
    wants ``.ico``; macOS wants ``.icns``, which needs ``iconutil`` and so is
    left to a macOS build.
    """
    try:
        from hairguard.tray import _make_icon_image

        image = _make_icon_image(256)
        target.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            path = target.with_suffix(".ico")
            image.save(
                path,
                sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (256, 256)],
            )
        else:
            path = target.with_suffix(".png")
            image.save(path)
        return path
    except Exception as exc:
        say(f"  значок не сгенерирован ({exc}) — сборка продолжится без него")
        return None


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def run_pyinstaller(models_dir: Path, icon: Path | None, clean: bool) -> int:
    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--log-level",
        "WARN",
        "--distpath",
        str(DIST_DIR),
        "--workpath",
        str(WORK_DIR),
        str(SPEC),
    ]
    if clean:
        command.insert(4, "--clean")

    environment = dict(os.environ)
    environment["HAIRGUARD_BUILD_MODELS"] = str(models_dir)
    if icon is not None:
        environment["HAIRGUARD_BUILD_ICON"] = str(icon)
    try:
        from hairguard import __version__

        environment["HAIRGUARD_BUILD_VERSION"] = __version__
    except Exception:
        pass

    say(f"  {' '.join(command[1:])}")
    return subprocess.call(command, cwd=str(REPO_ROOT), env=environment)


def bundle_root() -> Path:
    """Where the finished application landed."""
    app = DIST_DIR / f"{buildconf.APP_NAME}.app"
    if app.is_dir():
        return app
    return DIST_DIR / buildconf.APP_NAME


def console_binary(root: Path) -> Path | None:
    directory = root / "Contents" / "MacOS"
    if not directory.is_dir():
        directory = root
    for candidate in (
        directory / f"{buildconf.CONSOLE_NAME}.exe",
        directory / buildconf.CONSOLE_NAME,
    ):
        if candidate.is_file():
            return candidate
    return None


def interrogate_bundle(root: Path) -> list[str]:
    """Run the packaged ``doctor`` as a user who has installed nothing.

    ``HOME``/``USERPROFILE`` point at an empty directory, so the developer's
    own ``~/.hairguard/trained`` is out of reach and the answer has to come
    from what was bundled. Without that, the check would pass on this
    machine for the wrong reason and fail on everyone else's.

    This is also the only way to establish that ``hairlab.features`` made it
    in: pure Python lives in the bundle's archive rather than on disk, so
    walking the directory tree cannot tell.
    """
    binary = console_binary(root)
    if binary is None:
        return [
            f"Не найден {buildconf.CONSOLE_NAME} — упакованное приложение "
            "нечем спросить."
        ]

    with tempfile.TemporaryDirectory(prefix="hairguard-pristine-") as home:
        environment = dict(os.environ)
        environment["HOME"] = home
        environment["USERPROFILE"] = home
        environment["HAIRGUARD_LOG"] = str(Path(home) / "build-check.log")
        environment.pop("HOMEDRIVE", None)
        environment.pop("HOMEPATH", None)
        try:
            finished = subprocess.run(
                [str(binary), "doctor", "--json"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=environment,
                timeout=240,
            )
        except subprocess.TimeoutExpired:
            return ["Упакованный `doctor` не ответил за 240 с."]
        except OSError as exc:
            return [f"Упакованный `doctor` не запустился: {exc}"]

        output = (finished.stdout or "").strip()
        stderr = (finished.stderr or "").strip()
        # Tolerant of a stray line before the object: a build failing with
        # "did not return JSON" because something printed a warning would
        # hide whatever the report actually said.
        start = output.find("{")
        try:
            report = json.loads(output[start:] if start >= 0 else output)
        except ValueError:
            for line in (output + "\n" + stderr).splitlines()[:40]:
                if line.strip():
                    say(f"  | {line}")
            return [
                f"Упакованный `doctor` не вернул JSON (код {finished.returncode})."
            ]

        say(f"  | версия:      {report.get('version')}")
        say(f"  | упакованная: {report.get('frozen')}")
        say(f"  | признаков:   {report.get('feature_dim')}")
        say(f"  | onnxruntime: {report.get('onnxruntime')}")
        say(f"  | детектор:    {report.get('detector')}")
        say(
            "  | обучающий стек: "
            + (", ".join(report.get("training_stack") or []) or "отсутствует")
        )
        for name, where in (report.get("landmark_models") or {}).items():
            say(f"  | {name:<28} {where}")

        problems: list[str] = []
        if not report.get("frozen"):
            problems.append("Упакованное приложение не считает себя упакованным.")
        if not report.get("features_ok"):
            problems.append(
                "В сборке нет hairlab.features — модель будет отвергнута при "
                "загрузке, и приложение останется на правиле."
            )
        if "модель" not in str(report.get("detector", "")):
            problems.append(
                "На чистом профиле приложение выбирает не модель, а: "
                f"{report.get('detector')}. Именно так выглядит сборка, "
                "которая у пользователя работает вполсилы."
            )
        for name, where in (report.get("landmark_models") or {}).items():
            if "скачается" in str(where):
                problems.append(
                    f"{name} не попал в сборку — при первом запуске "
                    "приложение будет молча его скачивать."
                )
        problems.extend(report.get("problems") or [])
        return problems


def add_platform_extras(root: Path, icon: Path | None) -> list[str]:
    """Drop in whatever the platform needs beyond the binaries themselves.

    Linux has no equivalent of a .app or an .exe association: a folder with
    an executable in it does not appear in the applications menu until
    something writes a .desktop entry, and that entry has to name the path
    the user unpacked to. So a small script goes in the folder to write it.
    """
    added: list[str] = []
    if sys.platform == "win32" or sys.platform == "darwin":
        return added

    source = Path(__file__).resolve().parent / "linux_install.sh"
    if source.is_file():
        target = root / "install-desktop-entry.sh"
        shutil.copyfile(source, target)
        target.chmod(0o755)
        added.append(target.name)
    if icon is not None and icon.suffix == ".png":
        shutil.copyfile(icon, root / "hairguard.png")
        added.append("hairguard.png")
    return added


def zip_bundle(root: Path) -> Path | None:
    """One file to hand over. The folder inside it is the application."""
    try:
        from hairguard import __version__

        version = __version__
    except Exception:
        version = "0.1.0"
    label = {"win32": "windows", "darwin": "macos"}.get(sys.platform, "linux")
    archive = DIST_DIR / f"{buildconf.APP_NAME}-{version}-{label}"
    say(f"  упаковываю в {archive.with_suffix('.zip').name}…")
    made = shutil.make_archive(
        str(archive), "zip", root_dir=str(root.parent), base_dir=root.name
    )
    return Path(made)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    parser.add_argument(
        "--models",
        type=Path,
        default=Path.home() / ".hairguard" / "models",
        help="Where the MediaPipe .task files are cached",
    )
    parser.add_argument(
        "--no-landmark-models",
        action="store_true",
        help="Build without bundling the MediaPipe models (first run will download)",
    )
    parser.add_argument(
        "--clean", action="store_true", help="Discard PyInstaller's cache first"
    )
    parser.add_argument(
        "--no-zip", action="store_true", help="Leave the folder, do not archive it"
    )
    parser.add_argument(
        "--skip-run-check",
        action="store_true",
        help="Do not launch the built binary to ask what it can see",
    )
    args = parser.parse_args(argv)

    started = time.monotonic()
    say(f"HairGuard · сборка для {sys.platform}")

    rule("проверка исходников")
    problems = buildconf.preflight(
        REPO_ROOT, args.models, require_landmark_models=not args.no_landmark_models
    )
    if problems:
        for problem in problems:
            say(f"  FAIL: {problem}")
        say()
        say("Сборка отменена. Собрать такое приложение хуже, чем не собрать:")
        say("оно запустилось бы и выглядело бы работающим.")
        return 1
    say("  ok: модель, извлекатель признаков, onnxruntime и MediaPipe на месте")

    rule("значок")
    icon = make_icon(WORK_DIR / "icon" / "hairguard")
    if icon is not None:
        say(f"  ok: {icon}")

    rule("pyinstaller")
    models = args.models if not args.no_landmark_models else Path("/nonexistent")
    code = run_pyinstaller(models, icon, args.clean)
    if code != 0:
        say(f"  FAIL: pyinstaller завершился с кодом {code}")
        return code

    root = bundle_root()
    if not root.exists():
        say(f"  FAIL: ожидал сборку в {root}, её там нет")
        return 1
    say(f"  ok: {root}")

    rule("проверка сборки")
    problems = buildconf.verify_bundle(root)
    if problems:
        for problem in problems:
            say(f"  FAIL: {problem}")
        return 1
    say("  ok: ничего из обучающего стека не попало внутрь")
    say("  ok: модель и оба исполняемых файла на месте")

    if not args.skip_run_check:
        rule("что скажет само приложение, на чистом профиле")
        problems = interrogate_bundle(root)
        if problems:
            for problem in problems:
                say(f"  FAIL: {problem}")
            return 1
        say("  ok: на машине без ~/.hairguard приложение выбирает модель,")
        say("      видит hairlab.features и все три модели MediaPipe")

    extras = add_platform_extras(root, icon)
    if extras:
        rule("для этой платформы")
        for name in extras:
            say(f"  ok: {name}")

    rule("размер")
    total = buildconf.total_size(root)
    say(f"  всего: {human(total)}")
    for name, size in buildconf.largest_components(root, limit=10):
        share = 100.0 * size / total if total else 0.0
        say(f"    {human(size):>10}  {share:4.1f}%  {name}")

    if not args.no_zip:
        rule("архив")
        archive = zip_bundle(root)
        if archive is not None:
            say(f"  ok: {archive}  ({human(archive.stat().st_size)})")

    rule("готово")
    say(f"  за {time.monotonic() - started:.0f} с")
    say(f"  запустить: {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
