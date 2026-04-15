import argparse
import sys
import time


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
    # Stage 6 will wire this to HairGuardApp.
    # For now we accept --camera and --background but do nothing.
    print("Not yet implemented")


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
                print(f"Не удалось завершить процесс {pid}.")
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
            import os

            with open(f"/proc/{pid}/stat") as f:
                fields = f.read().split()
            # Field 22 (0-indexed 21) = starttime in clock ticks since boot.
            starttime_ticks = int(fields[21])
            clk_tck = os.sysconf("SC_CLK_TCK")
            with open("/proc/uptime") as f:
                uptime_sec = float(f.read().split()[0])
            elapsed = uptime_sec - starttime_ticks / clk_tck

        return _fmt_elapsed(elapsed)
    except Exception:
        return ""


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
    p_start.set_defaults(func=cmd_start)

    # stop
    p_stop = sub.add_parser("stop", help="Stop running instance")
    p_stop.set_defaults(func=cmd_stop)

    # status
    p_status = sub.add_parser("status", help="Show running status")
    p_status.set_defaults(func=cmd_status)

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
