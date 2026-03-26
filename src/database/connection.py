import sqlite3
import os
import logging
import threading
from datetime import date, datetime
from typing import List, Optional
from src.config import DB_PATH
from src.database.schema import create_tables, create_indexes, migrate_schema

logger = logging.getLogger(__name__)


def _adapt_date(value: date) -> str:
    return value.isoformat()


def _adapt_datetime(value: datetime) -> str:
    return value.isoformat(sep=" ")


def _convert_date(raw: bytes) -> date:
    return date.fromisoformat(raw.decode("utf-8"))


def _convert_timestamp(raw: bytes) -> datetime:
    text = raw.decode("utf-8")
    if text.endswith("Z"):
        text = f"{text[:-1]}+00:00"
    return datetime.fromisoformat(text)


sqlite3.register_adapter(date, _adapt_date)
sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("DATE", _convert_date)
sqlite3.register_converter("TIMESTAMP", _convert_timestamp)


class DBConnection:
    """Thread-safe SQLite connection manager using per-thread connections.

    Each thread gets its own ``sqlite3.Connection`` via ``threading.local()``.
    WAL journal mode allows concurrent readers with a single writer, which
    matches the typical workload (many background readers, occasional writes
    serialised by ``_write_lock``).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DB_PATH
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self._ensure_directory_exists()

        # Bootstrap: create/migrate tables on the main thread connection.
        try:
            main_conn = self._get_conn()
            self._tune_connection(main_conn)
            create_tables(main_conn, self._write_lock)
            migrate_schema(main_conn, self._write_lock)
            create_indexes(main_conn, self._write_lock)
            logger.info(f"[DATABASE] Connected to {self.db_path}")
        except sqlite3.Error as e:
            logger.critical(f"[DATABASE] Fatal connection error: {e}", exc_info=True)
            raise

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _ensure_directory_exists(self):
        directory = os.path.dirname(self.db_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=60.0,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._tune_connection(conn)
        return conn

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection, creating one if needed."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self._create_connection()
            self._local.conn = conn
        return conn

    @staticmethod
    def _tune_connection(conn: sqlite3.Connection):
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=60000;")
            cursor.execute("PRAGMA cache_size=-64000;")
            cursor.execute("PRAGMA temp_store=MEMORY;")
            cursor.execute("PRAGMA mmap_size=268435456;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            conn.commit()
            cursor.close()
        except sqlite3.Error as e:
            logger.warning(f"[DATABASE] Tuning warning: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_cursor(self) -> sqlite3.Cursor:
        return self._get_conn().cursor()

    def get_connection(self) -> sqlite3.Connection:
        return self._get_conn()

    def execute_query(self, query: str, params: tuple = ()) -> List[sqlite3.Row]:
        conn = self._get_conn()
        cursor = conn.cursor()
        try:
            cursor.execute(query, params)
            return cursor.fetchall()
        except sqlite3.Error as e:
            logger.error(f"[DATABASE] Query Error: {e}")
            raise
        finally:
            cursor.close()

    def execute_update(self, query: str, params: tuple = ()) -> int:
        conn = self._get_conn()
        with self._write_lock:
            cursor = conn.cursor()
            try:
                cursor.execute(query, params)
                affected = cursor.rowcount
                conn.commit()
                return affected
            except sqlite3.Error as e:
                conn.rollback()
                logger.error(f"[DATABASE] Update Error: {e}")
                raise
            finally:
                cursor.close()

    def maintenance(self):
        conn = self._get_conn()
        with self._write_lock:
            try:
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                logger.info("[DATABASE] Maintenance complete")
            except sqlite3.Error as e:
                logger.error(f"[DATABASE] Maintenance failed: {e}")

    def commit(self):
        self._get_conn().commit()

    def rollback(self):
        self._get_conn().rollback()

    def close(self):
        """Close the calling thread's connection (if any)."""
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None

    def close_all(self):
        """Best-effort close for the current thread (used at shutdown)."""
        self.close()

    def __del__(self):
        self.close()
