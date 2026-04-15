import sqlite3
from pathlib import Path

HAIRGUARD_DIR = Path.home() / ".hairguard"
DEFAULT_DB_PATH = HAIRGUARD_DIR / "hairguard.db"

# Each migration is a list of SQL statements to execute.
# Index 0 = migration to reach version 1, index 1 = migration to version 2, etc.
_MIGRATIONS: list[list[str]] = [
    # Version 1 — initial schema
    [
        """
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        )
        """,
        "INSERT INTO schema_version (version) VALUES (0)",
        """
        CREATE TABLE IF NOT EXISTS episodes (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at   TEXT    NOT NULL,
            duration_sec REAL    NOT NULL
        )
        """,
    ],
]


class Database:
    """Manages the SQLite connection and schema migrations for HairGuard."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = DEFAULT_DB_PATH

        self._path = Path(db_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            str(self._path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row

        self._run_migrations()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_connection(self) -> sqlite3.Connection:
        """Return the underlying sqlite3 connection."""
        return self._conn

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ------------------------------------------------------------------
    # Migration runner
    # ------------------------------------------------------------------

    def _current_version(self) -> int:
        """Return the current schema version, or 0 if not yet initialised."""
        cursor = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        if cursor.fetchone() is None:
            return 0
        row = self._conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
        if row is None:
            return 0
        return int(row[0])

    def _run_migrations(self) -> None:
        """Apply any pending migrations sequentially."""
        version = self._current_version()
        target = len(_MIGRATIONS)

        for v in range(version, target):
            statements = _MIGRATIONS[v]
            with self._conn:
                for sql in statements:
                    self._conn.execute(sql)
                # Update the version counter after each migration succeeds.
                # For v==0 the row was just inserted with value 0; UPDATE it now.
                self._conn.execute("UPDATE schema_version SET version = ?", (v + 1,))
