"""Statistics recording and CLI for HairGuard."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Optional

import colorama
from colorama import Fore, Style

from hairguard.db import Database


def _configure_stdout_utf8() -> None:
    """Reconfigure stdout/stderr to UTF-8 on Windows if needed."""
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
            sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except AttributeError:
            pass


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    id: int
    started_at: datetime
    duration_sec: float


# ---------------------------------------------------------------------------
# StatsRecorder
# ---------------------------------------------------------------------------


class StatsRecorder:
    """Writes and reads episode records from the SQLite database."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def record_episode(self, started_at: datetime, duration_sec: float) -> None:
        """Insert a completed episode into the database."""
        conn = self._db.get_connection()
        with conn:
            conn.execute(
                "INSERT INTO episodes (started_at, duration_sec) VALUES (?, ?)",
                (started_at.isoformat(), duration_sec),
            )

    def get_episodes(self, from_dt: datetime, to_dt: datetime) -> list[Episode]:
        """Return all episodes whose started_at falls within [from_dt, to_dt]."""
        conn = self._db.get_connection()
        cursor = conn.execute(
            """
            SELECT id, started_at, duration_sec
            FROM episodes
            WHERE started_at >= ? AND started_at <= ?
            ORDER BY started_at
            """,
            (from_dt.isoformat(), to_dt.isoformat()),
        )
        rows = cursor.fetchall()
        result: list[Episode] = []
        for row in rows:
            started = datetime.fromisoformat(row["started_at"])
            result.append(Episode(id=row["id"], started_at=started, duration_sec=row["duration_sec"]))
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _day_range(d: date) -> tuple[datetime, datetime]:
    """Return the start and end datetime for an entire day."""
    start = datetime(d.year, d.month, d.day, 0, 0, 0)
    end = datetime(d.year, d.month, d.day, 23, 59, 59, 999999)
    return start, end


def _parse_date(value: str) -> date:
    """Parse a date string: 'today', 'yesterday', or YYYY-MM-DD."""
    v = value.strip().lower()
    if v == "today":
        return date.today()
    if v == "yesterday":
        return date.today() - timedelta(days=1)
    return date.fromisoformat(value)


def _parse_period(period: str) -> int:
    """Parse a period string like '7d', '2w', '1m' and return number of days."""
    m = re.fullmatch(r"(\d+)([dwm])", period.strip().lower())
    if not m:
        raise ValueError(f"Invalid period '{period}'. Use e.g. 7d, 2w, 1m.")
    n = int(m.group(1))
    unit = m.group(2)
    if unit == "d":
        return n
    if unit == "w":
        return n * 7
    if unit == "m":
        return n * 30
    raise ValueError(f"Unknown unit '{unit}'")


def _fmt_duration(total_sec: float) -> str:
    """Format a duration in seconds as 'Xm Ys' or 'Ys'."""
    total_sec = int(total_sec)
    minutes, secs = divmod(total_sec, 60)
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def _histogram(episodes: list[Episode]) -> list[tuple[int, int]]:
    """Return (hour, count) pairs for each hour that has episodes."""
    counts: dict[int, int] = {}
    for ep in episodes:
        h = ep.started_at.hour
        counts[h] = counts.get(h, 0) + 1
    return sorted(counts.items())


def _render_bar(count: int, max_count: int, width: int = 10) -> str:
    """Render a Unicode block bar scaled to width characters."""
    if max_count == 0:
        filled = 0
    else:
        filled = round(count / max_count * width)
    return "█" * filled + "░" * (width - filled)


def _print_stats_block(
    episodes: list[Episode],
    hour_filter: Optional[int] = None,
    label: Optional[str] = None,
) -> None:
    """Print the standard stats block: counts, totals, histogram.

    When hour_filter is set, summary statistics are also restricted to
    episodes within that hour.
    """
    if label:
        print(f"\n{Style.BRIGHT}{label}{Style.RESET_ALL}")

    if hour_filter is not None:
        displayed = [ep for ep in episodes if ep.started_at.hour == hour_filter]
    else:
        displayed = episodes

    count = len(displayed)
    total_sec = sum(ep.duration_sec for ep in displayed)
    avg_sec = (total_sec / count) if count else 0.0

    print(f"  Эпизоды:      {count}")
    print(f"  Общее время:   {_fmt_duration(total_sec)}")
    print(f"  Среднее:       {_fmt_duration(avg_sec)}")

    hist = _histogram(displayed)
    if hist:
        max_count = max(c for _, c in hist)
        print("\n  По часам:")
        for hour, cnt in hist:
            bar = _render_bar(cnt, max_count)
            print(f"  {hour:02d}:00  {bar}  {cnt}")


def _print_comparison(
    episodes_a: list[Episode],
    episodes_b: list[Episode],
    label_a: str,
    label_b: str,
) -> None:
    """Print two periods side by side with a trend indicator."""
    count_a = len(episodes_a)
    count_b = len(episodes_b)
    total_a = sum(ep.duration_sec for ep in episodes_a)
    total_b = sum(ep.duration_sec for ep in episodes_b)
    avg_a = (total_a / count_a) if count_a else 0.0
    avg_b = (total_b / count_b) if count_b else 0.0

    col_w = 20
    print(f"\n  {'':20s}{Style.BRIGHT}{label_a:<{col_w}}{label_b}{Style.RESET_ALL}")
    print(f"  {'Эпизоды:':<20s}{count_a:<{col_w}}{count_b}")
    print(f"  {'Общее время:':<20s}{_fmt_duration(total_a):<{col_w}}{_fmt_duration(total_b)}")
    print(f"  {'Среднее:':<20s}{_fmt_duration(avg_a):<{col_w}}{_fmt_duration(avg_b)}")

    # Trend: compare count_b (current / second half) vs count_a (previous / first half)
    if count_a == 0 and count_b == 0:
        print(f"\n  Тренд: нет данных")
        return

    if count_a == 0:
        # No baseline — can't compute percentage
        print(f"\n  Тренд: нет базы для сравнения")
        return

    pct_change = (count_b - count_a) / count_a * 100
    abs_pct = abs(int(pct_change))

    if count_b < count_a:
        arrow = "↓"
        color = Fore.GREEN
        suffix = " 👍"
    elif count_b > count_a:
        arrow = "↑"
        color = Fore.RED
        suffix = ""
    else:
        arrow = "→"
        color = Style.RESET_ALL
        suffix = ""

    trend_str = f"{arrow} {abs_pct}% эпизодов{suffix}"
    print(f"\n  Тренд: {color}{trend_str}{Style.RESET_ALL}")


# ---------------------------------------------------------------------------
# StatsCLI
# ---------------------------------------------------------------------------


class StatsCLI:
    """Handles the 'hairguard stats' subcommand."""

    def __init__(self, db: Database) -> None:
        self._db = db
        self._recorder = StatsRecorder(db)

    def run(self, args: argparse.Namespace) -> None:
        _configure_stdout_utf8()
        colorama.init()

        try:
            self._dispatch(args)
        except ValueError as exc:
            print(f"Ошибка: {exc}")
            raise SystemExit(1)

    # ------------------------------------------------------------------
    # Dispatch logic
    # ------------------------------------------------------------------

    def _dispatch(self, args: argparse.Namespace) -> None:
        compare: bool = getattr(args, "compare", False)
        date_str: Optional[str] = getattr(args, "date", None)
        from_date_str: Optional[str] = getattr(args, "from_date", None)
        to_date_str: Optional[str] = getattr(args, "to_date", None)
        last_period: Optional[str] = getattr(args, "last", None)
        hour_filter: Optional[int] = getattr(args, "hour", None)

        # ---- --compare --last PERIOD ----
        if compare and last_period:
            self._show_comparison_period(last_period, hour_filter)
            return

        # ---- --from DATE --to DATE ----
        if from_date_str and to_date_str:
            from_dt = datetime.combine(_parse_date(from_date_str), datetime.min.time())
            to_dt = datetime.combine(_parse_date(to_date_str), datetime.max.time()).replace(microsecond=999999)
            episodes = self._recorder.get_episodes(from_dt, to_dt)
            label = f"{from_date_str} — {to_date_str}"
            _print_stats_block(episodes, hour_filter, label)
            return

        # ---- --last PERIOD ----
        if last_period:
            days = _parse_period(last_period)
            to_dt_l = datetime.combine(date.today(), datetime.max.time()).replace(microsecond=999999)
            from_dt_l = datetime.combine(date.today() - timedelta(days=days - 1), datetime.min.time())
            episodes = self._recorder.get_episodes(from_dt_l, to_dt_l)
            label = f"Последние {last_period}"
            _print_stats_block(episodes, hour_filter, label)
            return

        # ---- --date DATE ----
        if date_str:
            d = _parse_date(date_str)
            from_dt_d, to_dt_d = _day_range(d)
            episodes = self._recorder.get_episodes(from_dt_d, to_dt_d)
            label = str(d)
            _print_stats_block(episodes, hour_filter, label)
            return

        # ---- Default: today + comparison with yesterday ----
        self._show_today_vs_yesterday(hour_filter)

    # ------------------------------------------------------------------
    # High-level views
    # ------------------------------------------------------------------

    def _show_today_vs_yesterday(self, hour_filter: Optional[int]) -> None:
        today = date.today()
        yesterday = today - timedelta(days=1)

        from_today, to_today = _day_range(today)
        from_yesterday, to_yesterday = _day_range(yesterday)

        eps_today = self._recorder.get_episodes(from_today, to_today)
        eps_yesterday = self._recorder.get_episodes(from_yesterday, to_yesterday)

        _print_stats_block(eps_today, hour_filter, f"Сегодня ({today})")
        print()
        _print_comparison(
            eps_yesterday,
            eps_today,
            label_a=f"Вчера ({yesterday})",
            label_b=f"Сегодня ({today})",
        )

    def _show_comparison_period(self, period: str, hour_filter: Optional[int]) -> None:
        """Split the given period in half and compare both halves."""
        days = _parse_period(period)
        half = days // 2 or 1

        today = date.today()
        # Second half: most recent `half` days
        to_b = datetime.combine(today, datetime.max.time()).replace(microsecond=999999)
        from_b = datetime.combine(today - timedelta(days=half - 1), datetime.min.time())
        # First half: the `half` days before that
        to_a = datetime.combine(today - timedelta(days=half), datetime.max.time()).replace(microsecond=999999)
        from_a = datetime.combine(today - timedelta(days=half * 2 - 1), datetime.min.time())

        eps_a = self._recorder.get_episodes(from_a, to_a)
        eps_b = self._recorder.get_episodes(from_b, to_b)

        label_a = f"Первая половина ({half}д)"
        label_b = f"Вторая половина ({half}д)"

        _print_stats_block(eps_a, hour_filter, label_a)
        _print_stats_block(eps_b, hour_filter, label_b)
        print()
        _print_comparison(eps_a, eps_b, label_a=label_a, label_b=label_b)
