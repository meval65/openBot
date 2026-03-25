import logging
import threading
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Optional, Tuple
from uuid import uuid4
from src.database import DBConnection
from src.config import TIMEZONE

logger = logging.getLogger(__name__)

class SchedulerService:
    DUPLICATE_WINDOW = 5
    PAST_TOLERANCE_SECONDS = 90
    CLAIM_PREFIX = "__claim__:"
    CLAIM_STALE_SECONDS = 300
    
    def __init__(self, db: DBConnection):
        self.db = db
        self._lock = threading.RLock()

    def _get_now(self) -> datetime:
        try:
            tz = pytz.timezone(TIMEZONE)
        except Exception:
            tz = pytz.timezone("Asia/Jakarta")
        return datetime.now(tz).replace(tzinfo=None)

    def _normalize_context(self, context: str) -> str:
        return " ".join((context or "").split()).strip()

    def _normalize_trigger_time(self, trigger_time: datetime) -> Optional[datetime]:
        if not isinstance(trigger_time, datetime):
            return None
        try:
            if trigger_time.tzinfo is not None:
                try:
                    local_tz = pytz.timezone(TIMEZONE)
                except Exception:
                    local_tz = pytz.timezone("Asia/Jakarta")
                trigger_time = trigger_time.astimezone(local_tz)
            return trigger_time.replace(tzinfo=None)
        except Exception:
            return None

    def _validate_schedule_input(self, context: str, trigger_time: datetime) -> bool:
        normalized = self._normalize_trigger_time(trigger_time)
        now = self._get_now()
        return (
            context and 
            len(context.strip()) >= 3 and 
            normalized is not None and
            normalized >= (now - timedelta(seconds=self.PAST_TOLERANCE_SECONDS))
        )

    def _get_time_window(self, target_time: datetime, minutes: int) -> Tuple[datetime, datetime]:
        return (
            target_time - timedelta(minutes=minutes),
            target_time + timedelta(minutes=minutes)
        )

    def _canonical_context(self, context: str) -> str:
        return self._normalize_context(context).lower()

    @staticmethod
    def _in_placeholders(count: int) -> str:
        return ",".join("?" for _ in range(max(0, int(count))))

    def _claim_like(self) -> str:
        return f"{self.CLAIM_PREFIX}%"

    def _make_claim_note(self, owner: str) -> str:
        safe_owner = self._normalize_context(owner or "worker").replace(" ", "_").lower()
        return f"{self.CLAIM_PREFIX}{safe_owner}:{uuid4().hex}"

    def _claimable_sql(self) -> str:
        return (
            "("
            "execution_note IS NULL OR execution_note = '' "
            "OR execution_note NOT LIKE ? "
            "OR (execution_note LIKE ? AND (executed_at IS NULL OR executed_at <= ?))"
            ")"
        )

    def _claimable_params(self, stale_before: datetime) -> tuple:
        like = self._claim_like()
        return (like, like, stale_before)

    def _check_duplicate(self, cursor, canonical_context: str, start: datetime, end: datetime) -> bool:
        cursor.execute("""
            SELECT id FROM schedules 
            WHERE LOWER(TRIM(context)) = ? AND status = 'pending'
            AND scheduled_at BETWEEN ? AND ?
        """, (canonical_context, start, end))
        return cursor.fetchone() is not None

    def _find_existing_schedule(self, cursor, start: datetime, end: datetime) -> Optional[Tuple[int, str]]:
        cursor.execute("""
            SELECT id, context FROM schedules 
            WHERE status = 'pending'
            AND scheduled_at BETWEEN ? AND ?
            ORDER BY priority DESC, scheduled_at ASC
            LIMIT 1
        """, (start, end))
        row = cursor.fetchone()
        return (row[0], row[1]) if row else None

    def _context_similarity(self, a: str, b: str) -> float:
        a_tokens = set(self._canonical_context(a).split())
        b_tokens = set(self._canonical_context(b).split())
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        return float(inter / max(1, union))

    def _merge_schedule(self, cursor, schedule_id: int, old_context: str, new_context: str, priority: int) -> Optional[int]:
        old_clean = self._normalize_context(old_context)
        new_clean = self._normalize_context(new_context)
        if not new_clean:
            combined_context = old_clean
        else:
            parts = []
            seen = set()
            for part in (old_clean.split(" & ") + new_clean.split(" & ")):
                cleaned = self._normalize_context(part)
                if not cleaned:
                    continue
                key = cleaned.lower()
                if key in seen:
                    continue
                seen.add(key)
                parts.append(cleaned)
            combined_context = " & ".join(parts[:8])
            if len(combined_context) > 600:
                combined_context = combined_context[:600].rstrip()
        try:
            cursor.execute("""
                UPDATE schedules 
                SET context = ?, priority = MAX(priority, ?)
                WHERE id = ?
            """, (combined_context, priority, schedule_id))
            self.db.commit()
            logger.info(f"[SCHEDULER] Merged schedule ID {schedule_id}: {combined_context}")
            return schedule_id
        except Exception as e:
            logger.error(f"[SCHEDULER] Merge failed: {e}")
            return None

    def _insert_schedule(self, cursor, trigger_time: datetime, context: str, priority: int) -> Optional[int]:
        try:
            cursor.execute("""
                INSERT INTO schedules (scheduled_at, context, status, priority, created_at)
                VALUES (?, ?, 'pending', ?, ?)
            """, (trigger_time, context, priority, self._get_now()))
            self.db.commit()
            return cursor.lastrowid
        except Exception as e:
            logger.error(f"[SCHEDULER] Insert failed: {e}")
            return None

    def add_schedule(self, trigger_time: datetime, context: str, priority: int = 0) -> Optional[int]:
        context = self._normalize_context(context)
        trigger_time = self._normalize_trigger_time(trigger_time)

        if trigger_time is None or not self._validate_schedule_input(context, trigger_time):
            logger.warning(
                "[SCHEDULER] Rejected schedule input | context_len=%d trigger_time=%s now=%s",
                len(context or ""),
                str(trigger_time),
                str(self._get_now()),
            )
            return None

        with self._lock:
            cursor = self.db.get_cursor()
            priority = int(max(0, min(10, priority)))
            now = self._get_now()
            # If time is only slightly in the past (tool latency / processing delay),
            # snap it forward so near-term reminders (e.g. +30s) still get created.
            if trigger_time < now:
                trigger_time = now + timedelta(seconds=5)
            start, end = self._get_time_window(trigger_time, self.DUPLICATE_WINDOW)
            canonical = self._canonical_context(context)
            
            if self._check_duplicate(cursor, canonical, start, end):
                logger.info(f"[SCHEDULER] Duplicate blocked: {context} at {trigger_time}")
                return None

            existing = self._find_existing_schedule(cursor, start, end)
            
            if existing:
                schedule_id, old_context = existing
                # Avoid merging unrelated reminders that happen to be close in time.
                if self._context_similarity(old_context, context) >= 0.35:
                    return self._merge_schedule(cursor, schedule_id, old_context, context, priority)
                return self._insert_schedule(cursor, trigger_time, context, priority)

            return self._insert_schedule(cursor, trigger_time, context, priority)

    def cancel_schedule(self, schedule_id: int) -> bool:
        """
        Cancel a pending schedule by ID.
        Returns True if a row was actually updated, False otherwise.
        Encapsulates all DB access so callers never need to know the schema.
        """
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                cursor.execute(
                    "UPDATE schedules SET status='cancelled' WHERE id=? AND status='pending'",
                    (schedule_id,),
                )
                changed = cursor.rowcount > 0
                if changed:
                    self.db.commit()
                    logger.info(f"[SCHEDULER] Cancelled schedule ID {schedule_id}")
                else:
                    logger.warning(f"[SCHEDULER] cancel_schedule: ID {schedule_id} not found or already non-pending")
                return changed
            except Exception as e:
                logger.error(f"[SCHEDULER] cancel_schedule failed: {e}")
                return False



    def get_due_schedules(self, max_results: int = 20) -> List[Dict]:
        # Fetch all overdue pending schedules (<= now) so failed deliveries can retry.
        with self._lock:
            cursor = self.db.get_cursor()
            now = self._get_now()
            stale_before = now - timedelta(seconds=self.CLAIM_STALE_SECONDS)
            cursor.execute("""
                SELECT id, CAST(scheduled_at AS TEXT), context, priority
                FROM schedules
                WHERE status = 'pending' AND scheduled_at <= ?
                  AND (
                    execution_note IS NULL OR execution_note = ''
                    OR execution_note NOT LIKE ?
                    OR (execution_note LIKE ? AND (executed_at IS NULL OR executed_at <= ?))
                  )
                ORDER BY priority DESC, scheduled_at ASC
                LIMIT ?
            """, (now, self._claim_like(), self._claim_like(), stale_before, max_results))
            
            return [
                {
                    "id": row[0], 
                    "scheduled_at": row[1],
                    "context": row[2], 
                    "priority": row[3]
                }
                for row in cursor.fetchall()
            ]

    def get_pending_schedules(
        self,
        lookahead_minutes: int = 2,
        include_overdue: bool = True,
        max_results: int = 200,
    ) -> List[Dict]:
        with self._lock:
            cursor = self.db.get_cursor()
            now = self._get_now()
            stale_before = now - timedelta(seconds=self.CLAIM_STALE_SECONDS)
            target = now + timedelta(minutes=lookahead_minutes)
            limit = int(max(1, max_results))

            if include_overdue:
                cursor.execute("""
                    SELECT id, CAST(scheduled_at AS TEXT), context, priority
                    FROM schedules
                    WHERE status = 'pending' AND scheduled_at <= ?
                      AND (
                        execution_note IS NULL OR execution_note = ''
                        OR execution_note NOT LIKE ?
                        OR (execution_note LIKE ? AND (executed_at IS NULL OR executed_at <= ?))
                      )
                    ORDER BY priority DESC, scheduled_at ASC
                    LIMIT ?
                """, (target, self._claim_like(), self._claim_like(), stale_before, limit))
            else:
                cursor.execute("""
                    SELECT id, CAST(scheduled_at AS TEXT), context, priority
                    FROM schedules
                    WHERE status = 'pending' AND scheduled_at >= ? AND scheduled_at <= ?
                      AND (
                        execution_note IS NULL OR execution_note = ''
                        OR execution_note NOT LIKE ?
                        OR (execution_note LIKE ? AND (executed_at IS NULL OR executed_at <= ?))
                      )
                    ORDER BY priority DESC, scheduled_at ASC
                    LIMIT ?
                """, (now, target, self._claim_like(), self._claim_like(), stale_before, limit))
            
            return [
                {
                    "id": row[0], 
                    "scheduled_at": row[1],
                    "context": row[2], 
                    "priority": row[3]
                }
                for row in cursor.fetchall()
            ]


    def mark_as_executed(self, schedule_id: int, note: Optional[str] = None) -> bool:
        with self._lock:
            cursor = self.db.get_cursor()
            try:
                cursor.execute("""
                    UPDATE schedules 
                    SET status='executed', executed_at=?, execution_note=?
                    WHERE id=? AND status='pending'
                """, (self._get_now(), note, schedule_id))
                success = cursor.rowcount > 0
                if success:
                    self.db.commit()
                return success
            except Exception as e:
                logger.error(f"[SCHEDULER] Mark executed failed: {e}")
                return False

    def claim_pending_schedules(
        self,
        lookahead_minutes: int = 0,
        include_overdue: bool = True,
        max_results: int = 50,
        owner: str = "worker",
    ) -> Dict:
        with self._lock:
            cursor = self.db.get_cursor()
            now = self._get_now()
            stale_before = now - timedelta(seconds=self.CLAIM_STALE_SECONDS)
            target = now + timedelta(minutes=max(0, int(lookahead_minutes)))
            limit = int(max(1, max_results))
            claim_note = self._make_claim_note(owner)

            claimable_sql = self._claimable_sql()
            claimable_params = self._claimable_params(stale_before)

            if include_overdue:
                select_sql = f"""
                    SELECT id, CAST(scheduled_at AS TEXT), context, priority
                    FROM schedules
                    WHERE status = 'pending' AND scheduled_at <= ?
                      AND {claimable_sql}
                    ORDER BY priority DESC, scheduled_at ASC
                    LIMIT ?
                """
                select_params = (target, *claimable_params, limit)
            else:
                select_sql = f"""
                    SELECT id, CAST(scheduled_at AS TEXT), context, priority
                    FROM schedules
                    WHERE status = 'pending' AND scheduled_at >= ? AND scheduled_at <= ?
                      AND {claimable_sql}
                    ORDER BY priority DESC, scheduled_at ASC
                    LIMIT ?
                """
                select_params = (now, target, *claimable_params, limit)

            cursor.execute(select_sql, select_params)
            selected_rows = cursor.fetchall()
            if not selected_rows:
                return {"claim_note": "", "items": []}

            ids = [int(row[0]) for row in selected_rows]
            placeholders = self._in_placeholders(len(ids))
            if not placeholders:
                return {"claim_note": "", "items": []}

            update_sql = f"""
                UPDATE schedules
                SET execution_note = ?, executed_at = ?
                WHERE id IN ({placeholders})
                  AND status = 'pending'
                  AND {claimable_sql}
            """
            update_params = (claim_note, now, *ids, *claimable_params)
            cursor.execute(update_sql, update_params)
            if cursor.rowcount <= 0:
                self.db.rollback()
                return {"claim_note": "", "items": []}
            self.db.commit()

            # Re-read only rows claimed by this worker to handle race safely.
            fetch_sql = f"""
                SELECT id, CAST(scheduled_at AS TEXT), context, priority
                FROM schedules
                WHERE id IN ({placeholders})
                  AND status = 'pending'
                  AND execution_note = ?
                ORDER BY priority DESC, scheduled_at ASC
            """
            cursor.execute(fetch_sql, (*ids, claim_note))
            items = [
                {
                    "id": row[0],
                    "scheduled_at": row[1],
                    "context": row[2],
                    "priority": row[3],
                }
                for row in cursor.fetchall()
            ]
            return {"claim_note": claim_note if items else "", "items": items}

    def complete_claimed_as_executed(
        self,
        schedule_ids: List[int],
        claim_note: str,
        note: Optional[str] = None,
    ) -> int:
        sid_list = [int(sid) for sid in (schedule_ids or []) if sid is not None]
        if not sid_list or not str(claim_note or "").startswith(self.CLAIM_PREFIX):
            return 0
        with self._lock:
            cursor = self.db.get_cursor()
            placeholders = self._in_placeholders(len(sid_list))
            now = self._get_now()
            sql = f"""
                UPDATE schedules
                SET status='executed', executed_at=?, execution_note=?
                WHERE id IN ({placeholders})
                  AND status='pending'
                  AND execution_note=?
            """
            cursor.execute(sql, (now, note, *sid_list, claim_note))
            changed = int(cursor.rowcount or 0)
            if changed > 0:
                self.db.commit()
            return changed

    def release_claim(
        self,
        claim_note: str,
        schedule_ids: Optional[List[int]] = None,
    ) -> int:
        if not str(claim_note or "").startswith(self.CLAIM_PREFIX):
            return 0
        with self._lock:
            cursor = self.db.get_cursor()
            sid_list = [int(sid) for sid in (schedule_ids or []) if sid is not None]
            if sid_list:
                placeholders = self._in_placeholders(len(sid_list))
                sql = f"""
                    UPDATE schedules
                    SET execution_note=NULL, executed_at=NULL
                    WHERE id IN ({placeholders})
                      AND status='pending'
                      AND execution_note=?
                """
                cursor.execute(sql, (*sid_list, claim_note))
            else:
                cursor.execute("""
                    UPDATE schedules
                    SET execution_note=NULL, executed_at=NULL
                    WHERE status='pending' AND execution_note=?
                """, (claim_note,))
            changed = int(cursor.rowcount or 0)
            if changed > 0:
                self.db.commit()
            return changed

    def cleanup_old_schedules(self, days_old: int = 30) -> int:
        with self._lock:
            try:
                cutoff = self._get_now() - timedelta(days=days_old)
                cursor = self.db.get_cursor()
                cursor.execute("""
                    DELETE FROM schedules 
                    WHERE status IN ('executed', 'cancelled', 'failed') AND scheduled_at < ?
                """, (cutoff,))
                self.db.commit()
                return cursor.rowcount
            except Exception as e:
                logger.error(f"[SCHEDULER] Cleanup failed: {e}")
                return 0
