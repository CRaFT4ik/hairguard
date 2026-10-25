import argparse
import subprocess
import sys
import time
from typing import Optional, Sequence


def _pid_file():
    """The instance guard. A seam, so commands that consult it are testable."""
    from hairguard.pid import PidFile

    return PidFile()


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_cameras(_args: argparse.Namespace) -> None:
    from hairguard.camera import CameraManager

    cameras = CameraManager.list_cameras()
    if not cameras:
        print("No cameras found.")
        return
    for cam in cameras:
        print(f"  [{cam['index']}] {cam['name']}")


def cmd_config(args: argparse.Namespace) -> None:
    from hairguard.config import Config

    cfg = Config()
    if args.edit:
        cfg.open_editor()
    else:
        cfg.show()


def cmd_stats(args: argparse.Namespace) -> None:
    from hairguard.db import Database
    from hairguard.stats import StatsCLI

    db = Database()
    StatsCLI(db).run(args)


def cmd_start(args: argparse.Namespace) -> None:
    # --background: re-launch self as a detached background process
    if args.background:
        _start_background(args)
        return

    from hairguard.app import HairGuardApp

    # --collect / --no-collect set the shared switch before the app reads it;
    # without either flag the previous state carries over.
    collect = True if args.collect else (False if args.no_collect else None)
    app = HairGuardApp(camera_index=args.camera, background=False, collect=collect)
    app.run()


def _start_background(args: argparse.Namespace) -> None:
    """Re-launch 'hairguard start' (without --background) as a detached process.

    Inside a bundle ``sys.executable`` is the app rather than an interpreter,
    so the ``-m hairguard`` form would produce a command line the app's own
    argparse rejects.
    """
    if getattr(sys, "frozen", False):
        cmd = [sys.executable, "start"]
    else:
        cmd = [sys.executable, "-m", "hairguard", "start"]
    if args.camera is not None:
        cmd += ["--camera", str(args.camera)]
    if getattr(args, "collect", False):
        cmd += ["--collect"]
    if getattr(args, "no_collect", False):
        cmd += ["--no-collect"]

    if sys.platform == "win32":
        CREATE_NO_WINDOW = 0x08000000
        proc = subprocess.Popen(
            cmd,
            creationflags=CREATE_NO_WINDOW,
            close_fds=True,
        )
    else:
        proc = subprocess.Popen(
            cmd,
            start_new_session=True,
            close_fds=True,
        )

    print(f"HairGuard запущен в фоне (PID {proc.pid})")


def cmd_stop(_args: argparse.Namespace) -> None:
    from hairguard.pid import PidFile

    pid_file = PidFile()
    pid = pid_file.read()

    if pid is None:
        print("HairGuard не запущен (PID-файл не найден).")
        return

    if not pid_file.is_running():
        print(f"Найден устаревший PID-файл (PID {pid}). Удаляю.")
        pid_file.release()
        return

    print(f"Завершаю HairGuard (PID {pid})…")

    try:
        if sys.platform == "win32":
            import ctypes

            PROCESS_TERMINATE = 0x0001
            handle = ctypes.windll.kernel32.OpenProcess(PROCESS_TERMINATE, False, pid)
            if handle == 0:
                print(f"Не удалось открыть процесс {pid}.")
                return
            result = ctypes.windll.kernel32.TerminateProcess(handle, 0)
            ctypes.windll.kernel32.CloseHandle(handle)
            if not result:
                print(f"Не удалось завершить процесс {pid}. Удаляю PID-файл.")
                pid_file.release()
                return
        else:
            import signal

            import os

            os.kill(pid, signal.SIGTERM)
    except Exception as exc:
        print(f"Ошибка при завершении процесса: {exc}")
        return

    # Wait briefly for the process to disappear.
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if not pid_file.is_running():
            break
        time.sleep(0.2)

    if pid_file.is_running():
        print(f"Предупреждение: процесс {pid} всё ещё работает.")
    else:
        # Clean up a stale PID file if the app didn't remove it itself.
        if pid_file.read() == pid:
            pid_file.release()
        print("HairGuard остановлен.")


def cmd_collect(args: argparse.Namespace) -> None:
    """Start, stop or inspect dataset collection in the running app.

    The tray menu does the same thing, but a menu entry cannot be scripted
    and gives no way to check afterwards whether collection ever started.
    Both routes set the same file; the app adopts it within a second.
    """
    from hairguard.pid import PidFile

    try:
        from hairlab.clipstore import ClipStore, LabelStore
        from hairlab.state import CollectionState
    except ImportError:
        print("Сбор данных недоступен: пакет hairlab не найден.")
        return

    state = CollectionState()
    running = PidFile().is_running()

    if args.state == "status":
        print(f"Сбор данных: {'ВКЛЮЧЁН' if state.read() else 'выключен'}")
        print(f"HairGuard:   {'запущен' if running else 'не запущен'}")
        store = ClipStore()
        clips = store.clip_ids()
        labelled = len(LabelStore().labelled_ids())
        print(f"Клипов:      {len(clips)} (размечено {labelled})")
        print(f"Каталог:     {store.root}")
        if state.read() and not running:
            print("\nПереключатель включён, но приложение не работает — ничего не пишется.")
        return

    enabled = args.state == "on"
    state.write(enabled)
    print(f"Сбор данных {'включён' if enabled else 'выключен'}.")

    if not running:
        print("HairGuard не запущен — запустите 'hairguard start', запись начнётся сразу.")
    else:
        print("Приложение подхватит это в течение секунды.")


def cmd_model(_args: argparse.Namespace) -> None:
    """Report which detector the app will use, and why.

    The tray menu says the same thing, but a menu entry cannot be scripted
    and cannot be read before the app starts. A model that quietly failed to
    load looks exactly like a model that works badly, so there has to be a
    way to ask.
    """
    from hairguard.config import Config
    from hairguard.twistnet import DEFAULT_DIR, ModelUnavailable, TwistModel, load_configured

    config = Config()

    # What the app will do, first — that is the question being asked.
    _, status = load_configured(config)
    print(f"Детектор:   {status}")
    print(f"Каталог:    {DEFAULT_DIR}")

    # Then the artefacts themselves, loaded regardless of the switch: a user
    # who turned the model off still wants to know whether there is one.
    try:
        model = TwistModel.load()
    except ModelUnavailable as exc:
        print(f"Артефакты:  недоступны — {exc}")
        print(
            "\nПриложение работает на пороговом правиле: по собственным "
            "разметкам пользователя\nоно ловит 53.9% эпизодов. Модель ловит "
            "86.7% при той же частоте ложных\nсрабатываний. Обучить: "
            "python -m hairlab train"
        )
        return

    info = model.info
    min_frames = min(int(config.get("detection", "model", "min_frames") or info.seq_len),
                     info.seq_len)
    delay = float(config.get("detection", "model", "trigger_delay_sec") or 0.0)
    print(f"Признаков:  {info.feature_dim}")
    print(f"Окно:       {info.seq_len} кадров (~{info.seq_len / 6.75:.1f} с при 6.75 fps)")
    print(f"Порог:      {model.threshold:.4f} (из model.json)")
    print(f"Обучена:    {info.trained_at}")
    model.close()

    # The reaction time, spelled out. It is the number this app lives or dies
    # by, and it is not readable from the threshold or the window alone.
    print(
        f"\nСпрашиваем с {min_frames}-го кадра у головы "
        f"(~{(min_frames - 1) / 6.75:.2f} с), затем {delay:.2f} с подтверждения:"
    )
    print(f"  реакция ~{(min_frames - 1) / 6.75 + delay:.2f} с")
    print(
        "  по размеченным дням, не участвовавшим в обучении: 80.7% эпизодов, "
        "медиана 0.72 с,\n  ложные на 'просто держу' 6.9% — против 37.7%, "
        "1.20 с и 10.3% у правила.\n  Кривая: python -m hairlab truncation"
    )


def cmd_status(_args: argparse.Namespace) -> None:
    import os
    from hairguard.pid import PidFile

    pid_file = PidFile()
    pid = pid_file.read()

    if pid is None:
        print("HairGuard: не запущен")
        return

    if not pid_file.is_running():
        print(f"HairGuard: не запущен (устаревший PID-файл, PID {pid})")
        return

    # Report running state + PID.
    # Uptime is not stored by PidFile, so we try /proc/<pid>/stat on Linux.
    uptime_str = _get_uptime(pid)
    if uptime_str:
        print(f"HairGuard: запущен  •  PID {pid}  •  Время работы: {uptime_str}")
    else:
        print(f"HairGuard: запущен  •  PID {pid}")


def _get_uptime(pid: int) -> str:
    """Try to determine how long the process has been running.

    Returns a human-readable string or an empty string if unavailable.
    """
    try:
        if sys.platform == "win32":
            import ctypes
            import ctypes.wintypes as wintypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = ctypes.windll.kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, False, pid
            )
            if handle == 0:
                return ""

            creation = wintypes.FILETIME()
            exit_ = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            ok = ctypes.windll.kernel32.GetProcessTimes(
                handle,
                ctypes.byref(creation),
                ctypes.byref(exit_),
                ctypes.byref(kernel),
                ctypes.byref(user),
            )
            ctypes.windll.kernel32.CloseHandle(handle)
            if not ok:
                return ""

            # FILETIME is 100-nanosecond intervals since 1601-01-01.
            ft_val = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
            # Convert to Unix epoch: subtract 116444736000000000 (100-ns ticks
            # between 1601-01-01 and 1970-01-01) then divide by 10^7.
            EPOCH_DIFF = 116_444_736_000_000_000
            created_unix = (ft_val - EPOCH_DIFF) / 1e7
            elapsed = time.time() - created_unix
        else:
            elapsed = _unix_elapsed(pid)

        return _fmt_elapsed(elapsed)
    except Exception:
        return ""


def _unix_elapsed(pid: int) -> float:
    """Process elapsed seconds on Unix-like systems (Linux, macOS, BSD).

    Prefers /proc/<pid>/stat on Linux for speed; otherwise (or as a
    fallback) shells out to `ps -o etimes=` which is supported on macOS,
    BSD, and Linux and returns elapsed seconds directly.
    """
    import os

    if sys.platform.startswith("linux"):
        try:
            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
            starttime_ticks = int(fields[21])
            clk_tck = os.sysconf("SC_CLK_TCK")
            with open("/proc/uptime") as f:
                uptime_sec = float(f.read().split()[0])
            return uptime_sec - starttime_ticks / clk_tck
        except (FileNotFoundError, OSError, IndexError, ValueError):
            pass  # fall through to ps

    out = subprocess.check_output(
        ["ps", "-o", "etimes=", "-p", str(pid)],
        text=True,
        stderr=subprocess.DEVNULL,
    ).strip()
    return float(out)


def _fmt_elapsed(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 0:
        seconds = 0
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    parts = []
    if days:
        parts.append(f"{days}д")
    if hours:
        parts.append(f"{hours}ч")
    if minutes:
        parts.append(f"{minutes}м")
    parts.append(f"{secs}с")
    return " ".join(parts)


def cmd_autostart(args: argparse.Namespace) -> None:
    from hairguard.autostart import autostart_on, autostart_off

    if args.state == "on":
        autostart_on()
    else:
        autostart_off()


def cmd_debug(args: argparse.Namespace) -> Optional[int]:
    """Open the standalone debug view, which opens its own camera.

    Refused while the app is running, and that refusal is the point. This
    command builds a second ``CameraManager`` on a device the app already
    holds. On Windows that does not fail — it succeeds and returns black
    frames, which is how this project once had its own camera auto-detection
    report a live webcam as blank. The user would be looking at an empty
    picture and concluding the detector was broken.

    The tray's window has no camera at all; it is fed from the running
    detection tick. So the fix is to send people there rather than to make
    this command cleverer.
    """
    if not getattr(args, "force", False) and _pid_file().is_running():
        pid = _pid_file().read()
        print(
            f"HairGuard уже работает (PID {pid}) и держит камеру.\n"
            "\n"
            "Эта команда открыла бы камеру второй раз. В Windows это не "
            "ошибка: камера\n"
            "откроется и будет отдавать чёрные кадры — окно покажет пустоту, "
            "а виноват\n"
            "будет выглядеть детектор.\n"
            "\n"
            "Откройте окно диагностики из меню в трее: «Диагностика: открыть "
            "окно». Оно\n"
            "показывает то, что видит работающий детектор, и своей камеры не "
            "имеет.\n"
            "\n"
            "Если камер несколько и вторая свободна: hairguard debug --force "
            "--camera N"
        )
        return 1

    from hairguard.debugview import main as debugview_main

    debugview_main(camera=args.camera)
    return 0


def cmd_debug_viewer(_args: argparse.Namespace) -> Optional[int]:
    """Become the diagnostic window, reading frames from standard input.

    Not meant to be typed. The running app spawns this (see
    ``hairguard.debugfeed``) and pushes it the detector state it already
    computed, which is why this process has no camera and no detectors.
    """
    from hairguard.debugviewer import main as viewer_main

    return viewer_main()


def cmd_log(args: argparse.Namespace) -> None:
    """Show where the log is and what is at the end of it.

    The packaged app has no console, so every ``print`` it makes goes here.
    When someone says it does nothing, this is the command that answers.
    """
    from hairguard import diagnostics

    path = diagnostics.log_path()
    print(f"Журнал: {path}")

    limit = int(getattr(args, "tail", 40) or 40)
    lines = diagnostics.tail(limit)
    if not lines:
        print("Журнал пуст или не существует — приложение ещё не запускалось.")
        return
    print("-" * 70)
    for line in lines:
        print(line)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------
#
# Each probe is a small function so that the ones that cannot be simulated on
# a working machine can be tested by replacing them.


def _probe_features() -> tuple[bool, str]:
    """Is the feature extractor the model was trained with importable?

    The one failure that nothing else notices. ``twistnet`` compares the
    model's declared feature width against this module at load time, so
    without it the model is refused and the app runs the old rule.
    """
    try:
        from hairguard.labpath import ensure_hairlab_importable

        ensure_hairlab_importable()
        from hairlab.features import FEATURE_DIM

        return True, f"{FEATURE_DIM} признаков"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _probe_onnxruntime() -> tuple[bool, str]:
    try:
        import onnxruntime

        return True, onnxruntime.__version__
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


#: Modules that belong to training and must not be reachable from a shipped
#: build. torch alone is 2.8 GB.
_TRAINING_ONLY = ("torch", "pytorch_lightning", "torchmetrics", "sklearn", "onnx")


def _probe_training_stack() -> list[str]:
    """Which training-only modules this process can import.

    Asked from inside the running application because that is the only
    vantage point that settles it. Walking a bundle's directory tree finds
    torch, which is mostly compiled — but ``pytorch_lightning`` is pure
    Python and would live entirely inside the bundle's archive, invisible to
    any amount of looking at files. ``find_spec`` sees both.
    """
    import importlib.util

    found: list[str] = []
    for name in _TRAINING_ONLY:
        try:
            if importlib.util.find_spec(name) is not None:
                found.append(name)
        except (ImportError, ValueError):
            pass
    return found


def _load_config():
    from hairguard.config import Config

    return Config()


def _probe_landmark_models() -> list[tuple[str, str, bool]]:
    """``(filename, where it is, present)`` for each MediaPipe model."""
    from hairguard import assets
    from hairguard.detector import _MODEL_NAME as POSE
    from hairguard.face_detector import _MODEL_NAME as FACE
    from hairguard.hand_detector import _MODEL_NAME as HAND

    report: list[tuple[str, str, bool]] = []
    for name in (HAND, FACE, POSE):
        cached = assets.user_dir() / name
        bundled = assets.bundled_dir() / name
        if cached.is_file():
            report.append((name, f"в кэше ({cached.parent})", True))
        elif bundled.is_file():
            report.append((name, "в сборке", True))
        else:
            report.append((name, "нет — скачается при первом запуске", False))
    return report


def cmd_doctor(args: argparse.Namespace) -> Optional[int]:
    """Report what this installation can and cannot do, and why.

    The companion to the log. A packaged app has no console, so when it
    misbehaves there has to be something that can be run and read: the log
    says what happened, this says what is present. It is also what the build
    asks the finished bundle, from a home directory pointed somewhere empty,
    because pure Python modules live inside the bundle's archive and no
    amount of looking at the directory tree proves they are there.

    It deliberately never opens a camera. The moment it is most useful is
    while the app is running and holding the device, and a second open on
    Windows succeeds and returns black frames.
    """
    import json
    import sys as _sys

    from hairguard import __version__, diagnostics

    problems: list[str] = []
    report: dict = {
        "version": __version__,
        "frozen": diagnostics.is_frozen(),
        "executable": _sys.executable,
        "platform": _sys.platform,
        "log": str(diagnostics.log_path()),
    }

    features_ok, features_note = _probe_features()
    report["features_ok"] = features_ok
    if features_ok:
        report["feature_dim"] = int(features_note.split()[0])
    else:
        problems.append(
            f"hairlab.features недоступен ({features_note}) — модель будет "
            "отвергнута, приложение останется на пороговом правиле"
        )

    ort_ok, ort_note = _probe_onnxruntime()
    report["onnxruntime"] = ort_note if ort_ok else None
    if not ort_ok:
        problems.append(
            f"onnxruntime недоступен ({ort_note}) — запустить модель нечем, "
            "приложение останется на пороговом правиле"
        )

    config = None
    try:
        config = _load_config()
    except Exception as exc:
        problems.append(f"настройки не читаются: {exc}")

    detector_status = "неизвестно"
    if config is not None:
        try:
            from hairguard.twistnet import load_configured

            model, detector_status = load_configured(config)
            if model is not None:
                report["threshold"] = model.threshold
                report["seq_len"] = model.seq_len
                report["model_dir"] = str(model.info.directory)
                report["trained_at"] = model.info.trained_at
                model.close()
            elif "отключена" not in detector_status:
                # A deliberate `enabled: false` is the documented way back to
                # the rule and is not a fault. Anything else is.
                problems.append(f"модель не используется: {detector_status}")
        except Exception as exc:
            detector_status = f"не определён: {exc}"
            problems.append(f"не удалось выяснить детектор: {exc}")
    report["detector"] = detector_status

    landmarks = _probe_landmark_models()
    report["landmark_models"] = {name: where for name, where, _ in landmarks}

    camera_setting = None
    if config is not None:
        try:
            camera_setting = config.get("camera")
        except Exception:
            camera_setting = None
    report["camera"] = camera_setting

    running_pid = None
    try:
        guard = _pid_file()
        if guard.is_running():
            running_pid = guard.read()
    except Exception:
        running_pid = None
    report["running_pid"] = running_pid

    # Only meaningful inside a bundle, where importing torch would mean 2.8 GB
    # of it shipped. From a source checkout these are simply installed, which
    # is how the model got trained in the first place.
    unwanted = _probe_training_stack()
    report["training_stack"] = unwanted
    if report["frozen"] and unwanted:
        problems.append(
            "в сборку попал обучающий стек: " + ", ".join(unwanted) + " — "
            "приложению он не нужен, а torch один весит 2.8 ГБ"
        )

    report["problems"] = problems
    report["ok"] = not problems

    if getattr(args, "json", False):
        # Escaped rather than literal UTF-8, deliberately. Every message in
        # this report is Russian, and a Windows console hands the child
        # process a single-byte code page: the build's first attempt at
        # reading this back decoded cp1251 bytes as UTF-8 and concluded the
        # bundle had chosen the wrong detector. ASCII escapes survive any
        # code page, and ``json.loads`` turns them back into the real text.
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0 if not problems else 1

    print(f"HairGuard {__version__}")
    print(f"Сборка:      {'упакованная' if report['frozen'] else 'из исходников'}")
    print(f"Программа:   {report['executable']}")
    print(f"Платформа:   {report['platform']}")
    print(f"Журнал:      {report['log']}")
    print()
    print(
        f"hairlab.features: {'ok — ' + features_note if features_ok else 'НЕТ — ' + features_note}"
    )
    print(f"onnxruntime:      {ort_note if ort_ok else 'НЕТ — ' + ort_note}")
    if "model_dir" in report:
        print(
            f"Артефакты:        {report['model_dir']} "
            f"(порог {report['threshold']:.3f}, окно {report['seq_len']}, "
            f"обучена {report['trained_at']})"
        )
    print()
    print(f"Детектор: {detector_status}")
    print()
    print("Модели MediaPipe:")
    for name, where, _present in landmarks:
        print(f"  {name:<28} {where}")
    print()
    if report["frozen"]:
        print()
        print(
            "Обучающий стек: "
            + (", ".join(unwanted) + " — лишний вес" if unwanted else "отсутствует")
        )
    print()
    print(f"Камера:      {camera_setting!r} (не проверяется — её может держать приложение)")
    print(
        "Экземпляр:   "
        + (f"запущен, PID {running_pid}" if running_pid else "не запущен")
    )

    if problems:
        print()
        print(f"Проблем: {len(problems)}")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    return 0


def cmd_not_implemented(_args: argparse.Namespace) -> None:
    print("Not yet implemented")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hairguard",
        description="HairGuard — trichotillomania detection assistant",
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    sub.required = True

    # start
    p_start = sub.add_parser("start", help="Start detection")
    p_start.add_argument("--camera", type=int, default=None, metavar="INDEX")
    p_start.add_argument("--background", action="store_true")
    p_start.add_argument("--collect", action="store_true",
                         help="Record dataset clips from the start")
    p_start.add_argument("--no-collect", action="store_true",
                         help="Start without recording clips")
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = sub.add_parser("stop", help="Stop running instance")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show running status")
    p_status.set_defaults(func=cmd_status)

    # model
    p_model = sub.add_parser("model", help="Show which detector is in use")
    p_model.set_defaults(func=cmd_model)

    # collect
    p_collect = sub.add_parser("collect", help="Start/stop dataset collection")
    p_collect.add_argument("state", choices=["on", "off", "status"])
    p_collect.set_defaults(func=cmd_collect)

    # cameras
    p_cameras = sub.add_parser("cameras", help="List available cameras")
    p_cameras.set_defaults(func=cmd_cameras)

    # stats
    p_stats = sub.add_parser("stats", help="Show statistics")
    p_stats.add_argument("--date", metavar="DATE")
    p_stats.add_argument("--from", dest="from_date", metavar="DATE")
    p_stats.add_argument("--to", dest="to_date", metavar="DATE")
    p_stats.add_argument("--last", metavar="PERIOD")
    p_stats.add_argument("--hour", type=int, metavar="HOUR")
    p_stats.add_argument("--compare", action="store_true")
    p_stats.set_defaults(func=cmd_stats)

    # config
    p_config = sub.add_parser("config", help="Show or edit config")
    p_config.add_argument("--edit", action="store_true", help="Open in $EDITOR")
    p_config.set_defaults(func=cmd_config)

    # autostart
    p_autostart = sub.add_parser("autostart", help="Manage autostart")
    p_autostart.add_argument("state", choices=["on", "off"])
    p_autostart.set_defaults(func=cmd_autostart)

    # debug
    p_debug = sub.add_parser("debug", help="Open live debug/visualization view")
    p_debug.add_argument("--camera", type=int, default=None, metavar="N",
                         help="Camera index to use (default: from config)")
    p_debug.add_argument("--force", action="store_true",
                         help="Open a camera even while HairGuard is running")
    p_debug.set_defaults(func=cmd_debug)

    # debug-viewer — spawned by the running app, not typed by a person, so
    # it is left out of the command list. No `help=` at all rather than
    # help=SUPPRESS: argparse prints a subparser's help verbatim and happily
    # shows "==SUPPRESS==" in the listing.
    from hairguard.debugfeed import VIEWER_COMMAND

    p_viewer = sub.add_parser(VIEWER_COMMAND)
    p_viewer.set_defaults(func=cmd_debug_viewer)

    # doctor
    p_doctor = sub.add_parser(
        "doctor", help="Report what this installation can do, and what is missing"
    )
    p_doctor.add_argument("--json", action="store_true",
                          help="Machine-readable form, used by the build checks")
    p_doctor.set_defaults(func=cmd_doctor)

    # log
    p_log = sub.add_parser("log", help="Show the log file and its last lines")
    p_log.add_argument("--tail", type=int, default=40, metavar="N",
                       help="How many trailing lines to print (default 40)")
    p_log.set_defaults(func=cmd_log)

    return parser


def main(
    argv: Optional[Sequence[str]] = None,
    default_argv: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """Parse and dispatch.

    ``default_argv`` is what to do when given nothing, which only the
    windowed executable supplies: a double-click passes no arguments, and
    argparse's answer to no arguments is a usage error printed to a console
    that does not exist in that build.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments and default_argv:
        arguments = list(default_argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
