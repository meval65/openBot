import uuid
import logging
import numpy as np
import threading
import time
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from collections import defaultdict

from src.database.connection import DBConnection
from src.config import MemoryType, MAX_RETRIEVED_MEMORIES, DECAY_DAYS_EMOTION, DECAY_DAYS_GENERAL, DECAY_DAYS_MOOD

from src.services.memory.models import MemoryItem
from src.services.memory.embeddings import EmbeddingHandler
from src.services.memory.scorer import MemoryScorer

logger = logging.getLogger(__name__)

class MemoryManager:
    def __init__(self, db: DBConnection, similarity_threshold: float = 0.92):
        self.db = db
        self.similarity_threshold = similarity_threshold
        self._lock = threading.RLock()
        
        self.emb_handler = EmbeddingHandler()
        self.scorer = MemoryScorer()
        
        self._stats_cache = {}
        self._stats_cache_ts = 0.0
        self._stats_cache_ttl = 60.0
        self._batch_lock = threading.Lock()
        self._duplicate_check_limit = 64

    def add_memory(
        self,
        summary: str,
        m_type: str,
        priority: float = 0.5,
        embedding: List[float] = None,
        embedding_namespace: str = "memory",
    ):
        return self.add_memory_with_group(
            summary,
            m_type,
            priority,
            embedding,
            group_id=None,
            embedding_namespace=embedding_namespace,
        )

    @staticmethod
    def _normalize_memory_type(m_type: str) -> str:
        allowed = {m.value for m in MemoryType}
        normalized = str(m_type or "").strip().lower()
        return normalized if normalized in allowed else MemoryType.GENERAL.value

    @staticmethod
    def _normalize_embedding_namespace(namespace: Optional[str]) -> str:
        allowed = {"memory", "document", "image"}
        normalized = str(namespace or "").strip().lower()
        return normalized if normalized in allowed else "memory"

    @staticmethod
    def _has_embedding_payload(embedding) -> bool:
        if embedding is None:
            return False
        if isinstance(embedding, np.ndarray):
            return embedding.size > 0
        if isinstance(embedding, (list, tuple)):
            return len(embedding) > 0
        return True

    def add_memory_with_group(
        self,
        summary: str,
        m_type: str,
        priority: float = 0.5,
        embedding: List[float] = None,
        group_id: str = None,
        embedding_namespace: str = "memory",
    ):
        summary = " ".join((summary or "").split()).strip()
        if not summary or len(summary) < 3:
            logger.debug("[MEMORY-ADD] Invalid summary: too short")
            return "invalid"

        priority = max(0.0, min(1.0, priority))
        m_type = self._normalize_memory_type(m_type)
        embedding_namespace = self._normalize_embedding_namespace(embedding_namespace)
        
        with self._lock:
            vec_data = None
            embedding_blob = None
            
            if self._has_embedding_payload(embedding):
                result = self.emb_handler.prepare(embedding)
                if result:
                    vec_data, embedding_blob = result
            
            return self._add_memory_direct(
                summary, m_type, priority, vec_data, embedding_blob, group_id, embedding_namespace
            )

    def _add_memory_direct(self, summary: str, m_type: str, 
                           priority: float, vec_data: Optional[np.ndarray],
                           embedding_blob: Optional[bytes], group_id: str = None,
                           embedding_namespace: str = "memory"):
        if vec_data is not None:
            if self._check_duplicate(vec_data, summary, embedding_namespace):
                logger.info("[MEMORY-ADD] Duplicate detected, skipping")
                return "duplicate"
        
        mem_id = str(uuid.uuid4())
        cursor = self.db.get_cursor()
        
        try:
            cursor.execute("""
                INSERT INTO memories (id, summary, memory_type, embedding_namespace, priority, 
                                    last_used_at, use_count, status, embedding, created_at, group_id)
                VALUES (?, ?, ?, ?, ?, ?, 0, 'active', ?, ?, ?)
            """, (mem_id, summary, m_type, embedding_namespace, priority, datetime.now(), embedding_blob, datetime.now(), group_id))
            self.db.commit()
            self._invalidate_cache()
            logger.info(f"[MEMORY-ADD] Added memory (group={group_id}): {summary[:50]}")
            return "created"
        except Exception as e:
            self.db.rollback()
            logger.error(f"[MEMORY-ADD] {e}")
            return "failed"

    def update_memory(
        self,
        memory_id: str,
        new_summary: str,
        new_priority: Optional[float] = None,
        new_m_type: Optional[str] = None,
        new_embedding: Optional[List[float]] = None,
    ) -> bool:
        raw_id = str(memory_id or "").strip()
        if not raw_id:
            logger.warning("[MEMORY-UPDATE] Missing memory id")
            return False
        summary = " ".join((new_summary or "").split()).strip()
        if not summary or len(summary) < 3:
            logger.warning(f"[MEMORY-UPDATE] Invalid summary for {memory_id}")
            return False

        with self._lock:
            try:
                resolved_id = self._resolve_active_memory_id(raw_id)
                if not resolved_id:
                    logger.warning(f"[MEMORY-UPDATE] Target id not found/ambiguous: {raw_id}")
                    return False

                fields = ["summary=?", "last_used_at=?"]
                params: list = [summary, datetime.now()]

                if new_priority is not None:
                    fields.append("priority=?")
                    params.append(max(0.0, min(1.0, float(new_priority))))

                if new_m_type is not None:
                    fields.append("memory_type=?")
                    params.append(self._normalize_memory_type(new_m_type))

                if self._has_embedding_payload(new_embedding):
                    result = self.emb_handler.prepare(new_embedding)
                    if result:
                        _, embedding_blob = result
                        fields.append("embedding=?")
                        params.append(embedding_blob)

                params.append(resolved_id)
                sql = f"UPDATE memories SET {', '.join(fields)} WHERE id=? AND status='active'"

                cursor = self.db.get_cursor()
                cursor.execute(sql, params)
                if cursor.rowcount > 0:
                    self.db.commit()
                    self._invalidate_cache()
                    logger.info(f"[MEMORY-UPDATE] Updated memory {resolved_id}")
                    return True
                return False
            except Exception as e:
                self.db.rollback()
                logger.error(f"[MEMORY-UPDATE] {e}")
                return False

    def search_memories_by_text(
        self,
        query_text: str,
        max_results: int = 10,
        embedding_namespaces: Optional[List[str]] = None,
    ) -> List[Dict]:
        normalized_query = " ".join(str(query_text or "").lower().split()).strip()
        if not normalized_query:
            return []
        query_tokens = [t for t in dict.fromkeys(normalized_query.split()) if len(t) >= 2][:6]
        ns_list = embedding_namespaces or ["memory"]
        normalized_ns = [self._normalize_embedding_namespace(ns) for ns in ns_list]
        ns_placeholders = ",".join(["?"] * len(normalized_ns))
        token_clauses = " OR ".join(["lower(summary) LIKE ?"] * len(query_tokens))
        where_like = "lower(summary) LIKE ?"
        if token_clauses:
            where_like = f"({where_like} OR {token_clauses})"
        cursor = self.db.get_cursor()
        try:
            cursor.execute(
                f"""
                SELECT id, summary, memory_type, priority, embedding_namespace
                FROM memories
                WHERE status='active'
                  AND {where_like}
                  AND embedding_namespace IN ({ns_placeholders})
                ORDER BY priority DESC, last_used_at DESC, created_at DESC
                LIMIT ?
                """,
                [f"%{normalized_query}%"] + [f"%{t}%" for t in query_tokens] + normalized_ns + [max(1, int(max_results or 10))],
            )
            rows = cursor.fetchall()
            results = []
            ids_to_update = []
            for row in rows:
                results.append(
                    {
                        "id": row[0],
                        "summary": row[1],
                        "type": row[2],
                        "priority": row[3],
                        "use_count": 0,
                        "last_used": None,
                        "score": 0.0,
                        "embedding_namespace": row[4] or "memory",
                    }
                )
                ids_to_update.append(row[0])
            if ids_to_update:
                self._mark_used(ids_to_update)
            return results
        except Exception as e:
            logger.error(f"[MEMORY-TEXT-SEARCH] {e}")
            return []

    def _check_duplicate(self, vector: np.ndarray, summary: str, embedding_namespace: str = "memory") -> bool:
        cursor = self.db.get_cursor()
        normalized_summary = " ".join((summary or "").lower().split())
        
        try:
            cursor.execute(
                "SELECT 1 FROM memories WHERE lower(summary)=? AND status='active' AND embedding_namespace=? LIMIT 1",
                (normalized_summary, embedding_namespace)
            )
            if cursor.fetchone():
                return True
        except Exception as e:
            logger.error(f"[DUP-CHECK] Exact match check failed: {e}")

        try:
            cursor.execute("""
                SELECT id, embedding FROM memories 
                WHERE status='active' AND embedding IS NOT NULL
                  AND embedding_namespace = ?
                ORDER BY last_used_at DESC
                LIMIT ?
            """, (embedding_namespace, int(self._duplicate_check_limit)))
            
            rows = cursor.fetchall()
            if not rows:
                return False

            embeddings = []
            ids = []
            
            for r_id, r_emb in rows:
                vec = self.emb_handler.parse(r_emb)
                if vec is not None:
                    embeddings.append(vec)
                    ids.append(r_id)

            if not embeddings:
                return False

            similarities = self.emb_handler.compute_similarity_matrix(embeddings, vector)
            max_sim = np.max(similarities)
            
            if max_sim > self.similarity_threshold:
                best_idx = np.argmax(similarities)
                self._mark_used([ids[best_idx]])
                logger.info(f"[DUP-CHECK] Similar memory found (sim={max_sim:.3f})")
                return True
        except Exception as e:
            logger.error(f"[DUP-CHECK] Similarity check failed: {e}")

        return False


    def get_relevant_memories(
        self,
        query_embedding: List[float] = None,
        memory_type: str = None,
        min_priority: float = 0.0,
        max_results: int = None,
        embedding_namespaces: Optional[List[str]] = None,
    ) -> List[Dict]:
        max_results = max_results or MAX_RETRIEVED_MEMORIES
        return self._search_by_semantic(
            query_embedding, memory_type, min_priority, max_results, embedding_namespaces
        )


    def _search_by_semantic(
        self,
        query_embedding: List[float],
        memory_type: str,
        min_priority: float,
        max_results: int,
        embedding_namespaces: Optional[List[str]] = None,
    ) -> List[Dict]:
        items = self._fetch_memories(memory_type, min_priority, embedding_namespaces)
        
        if not items:
            return []
        
        query_vec = None
        if query_embedding is not None:
            if isinstance(query_embedding, np.ndarray) and query_embedding.size == 0:
                query_embedding = None
            elif isinstance(query_embedding, (list, tuple)) and len(query_embedding) == 0:
                query_embedding = None

        if query_embedding is not None:
            result = self.emb_handler.prepare(query_embedding)
            if result:
                query_vec, _ = result
        
        scores = self.scorer.calculate(items, query_vec, self.emb_handler)
        
        results = self._select_top_memories(
            items, scores, max_results, query_vec is not None
        )
        
        return results

    def _fetch_memories(
        self,
        memory_type: Optional[str],
        min_priority: float,
        embedding_namespaces: Optional[List[str]] = None,
    ) -> List[MemoryItem]:
        query = """
            SELECT id, summary, priority, embedding, use_count, last_used_at, memory_type, embedding_namespace
            FROM memories WHERE status = 'active'
        """
        params = []
        
        if memory_type:
            query += " AND memory_type = ?"
            params.append(memory_type)
        if min_priority > 0:
            query += " AND priority >= ?"
            params.append(min_priority)

        ns_list = embedding_namespaces or ["memory"]
        normalized_ns = [self._normalize_embedding_namespace(ns) for ns in ns_list]
        placeholders = ",".join(["?"] * len(normalized_ns))
        query += f" AND embedding_namespace IN ({placeholders})"
        params.extend(normalized_ns)
        
        query += " ORDER BY priority DESC, last_used_at DESC LIMIT 300"

        cursor = self.db.get_cursor()
        try:
            cursor.execute(query, params)
            
            items = []
            for row in cursor.fetchall():
                embedding = self.emb_handler.parse(row[3])
                items.append(MemoryItem(
                    id=row[0],
                    summary=row[1],
                    priority=row[2],
                    embedding=embedding,
                    use_count=row[4],
                    last_used=row[5],
                    memory_type=row[6],
                    embedding_namespace=row[7] or "memory",
                ))
            
            return items
        except Exception as e:
            logger.error(f"[FETCH-MEM] {e}")
            return []

    def _select_top_memories(self, items: List[MemoryItem], scores: np.ndarray,
                            max_results: int, is_semantic: bool) -> List[Dict]:
        if len(scores) == 0:
            return []
        
        sorted_indices = np.argsort(scores)[::-1]
        
        results = []
        ids_to_update = []
        type_counts = defaultdict(int)
        char_count = 0
        max_chars = 4000
        max_per_type = 5
        
        for idx in sorted_indices:
            if len(results) >= max_results:
                break
            
            score = float(scores[idx])
            if is_semantic and score < 0.3:
                continue

            item = items[idx]
            
            if char_count + len(item.summary) > max_chars:
                continue
            
            if type_counts[item.memory_type] >= max_per_type:
                continue

            results.append({
                'id': item.id,
                'summary': item.summary,
                'type': item.memory_type,
                'priority': item.priority,
                'use_count': item.use_count,
                'last_used': item.last_used,
                'score': score,
                'embedding_namespace': item.embedding_namespace,
            })
            
            ids_to_update.append(item.id)
            type_counts[item.memory_type] += 1
            char_count += len(item.summary)

        if ids_to_update:
            self._mark_used(ids_to_update)

        return results

    def _mark_used(self, memory_ids: List[str]):
        """Batched update of memory usage stats"""
        if not memory_ids:
            return
        
        try:
            now = datetime.now()
            
            BATCH_SIZE = 100
            for i in range(0, len(memory_ids), BATCH_SIZE):
                chunk = memory_ids[i:i + BATCH_SIZE]
                placeholders = ','.join(['?'] * len(chunk))

                self.db.execute_update(
                    f"""UPDATE memories 
                        SET use_count = use_count + 1, 
                            last_used_at = ? 
                        WHERE id IN ({placeholders})""",
                    tuple([now] + chunk),
                )
        except Exception as e:
            logger.error(f"[MARK-USED] Failed to update stats: {e}")

    def wipe_all_memories(self) -> int:
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                cursor.execute("DELETE FROM memories")
                count_memories = cursor.rowcount
                cursor.execute("DELETE FROM schedules")
                count = count_memories + cursor.rowcount
                self.db.commit()
                
                try:
                    conn = self.db.get_connection()
                    old_iso = conn.isolation_level
                    conn.isolation_level = None
                    conn.execute("VACUUM")
                    conn.isolation_level = old_iso
                except Exception as ve:
                    logger.warning(f"[WIPE] VACUUM failed: {ve}")
                    
                self._invalidate_cache()
                logger.info("[WIPE] All tables cleared successfully")
                return count
            except Exception as e:
                logger.error(f"[WIPE] Failed to clear all tables: {e}")
                self.db.rollback()
                return 0

    def archive_memory_by_id(self, memory_id: str) -> bool:
        """
        Archive (soft-delete) a single active memory by its UUID string.
        Returns True if a row was actually updated, False otherwise.
        Encapsulates all DB access — callers never need to know the schema.
        """
        raw_id = str(memory_id or "").strip()
        if not raw_id:
            return False
        with self._lock:
            try:
                resolved_id = self._resolve_active_memory_id(raw_id)
                if not resolved_id:
                    logger.warning(f"[MEMORY-ARCHIVE] ID not found/ambiguous: {raw_id}")
                    return False
                cursor = self.db.get_cursor()
                cursor.execute(
                    "UPDATE memories SET status='archived', priority=0, last_used_at=? "
                    "WHERE id=? AND status='active'",
                    (datetime.now(), resolved_id),
                )
                changed = cursor.rowcount > 0
                if changed:
                    self.db.commit()
                    self._invalidate_cache()
                    logger.info(f"[MEMORY-ARCHIVE] Archived by ID: {resolved_id[:8]}...")
                else:
                    logger.warning(f"[MEMORY-ARCHIVE] ID {resolved_id[:8]}... not found or already archived")
                return changed
            except Exception as e:
                logger.error(f"[MEMORY-ARCHIVE] Failed: {e}")
                return False

    def _resolve_active_memory_id(self, memory_id: str) -> Optional[str]:
        token = str(memory_id or "").strip()
        if not token:
            return None
        token = token.replace("...", "").strip()
        cursor = self.db.get_cursor()
        try:
            cursor.execute(
                "SELECT id FROM memories WHERE id=? AND status='active' LIMIT 1",
                (token,),
            )
            row = cursor.fetchone()
            if row:
                return row[0] if isinstance(row, (list, tuple)) else row["id"]

            # Accept short prefix IDs shown in UI/chat (e.g. 54667409...).
            if len(token) >= 6:
                cursor.execute(
                    "SELECT id FROM memories WHERE id LIKE ? AND status='active' LIMIT 2",
                    (f"{token}%",),
                )
                rows = cursor.fetchall()
                if len(rows) == 1:
                    only = rows[0]
                    return only[0] if isinstance(only, (list, tuple)) else only["id"]
        except Exception as e:
            logger.error(f"[MEMORY-ID-RESOLVE] Failed for '{token}': {e}")
        return None

    def forget_memory(self, query: str, embedding: List[float] = None) -> Optional[str]:
        with self._lock:
            query_text = " ".join(str(query or "").lower().split()).strip()
            if not query_text:
                return None

            cursor = self.db.get_cursor()

            # 1) Lexical-first deletion (more predictable for user-facing queries).
            try:
                cursor.execute(
                    """
                    SELECT id, summary
                    FROM memories
                    WHERE status='active'
                      AND embedding_namespace='memory'
                      AND lower(summary) LIKE ?
                    ORDER BY priority DESC, last_used_at DESC
                    LIMIT 1
                    """,
                    (f"%{query_text}%",),
                )
                matched = cursor.fetchone()
                if matched:
                    mem_id, mem_summary = matched[0], matched[1]
                    cursor.execute(
                        """
                        UPDATE memories
                        SET status='archived', priority=0, last_used_at=?
                        WHERE id=? AND status='active'
                        """,
                        (datetime.now(), mem_id),
                    )
                    if cursor.rowcount > 0:
                        self.db.commit()
                        self._invalidate_cache()
                        logger.info(f"[MEMORY-FORGET] Forgot by lexical match: {str(mem_summary)[:80]}")
                        return mem_summary
            except Exception as e:
                logger.error(f"[MEMORY-FORGET] Lexical match failed: {e}")

            query_vec = None
            if self._has_embedding_payload(embedding):
                res = self.emb_handler.prepare(embedding)
                if res:
                    query_vec, _ = res
            
            if query_vec is None:
                logger.warning(f"[MEMORY-FORGET] No embedding provided for query: {query}")
                return None

            try:
                cursor.execute("""
                    SELECT id, summary, embedding FROM memories 
                    WHERE status='active' AND embedding IS NOT NULL
                      AND embedding_namespace='memory'
                """)
                
                rows = cursor.fetchall()
                if not rows:
                    return None

                mem_ids = []
                mem_vecs = []
                mem_summaries = []

                for r in rows:
                    vec = self.emb_handler.parse(r[2])
                    if vec is not None:
                        mem_ids.append(r[0])
                        mem_summaries.append(r[1])
                        mem_vecs.append(vec)

                if not mem_vecs:
                    return None

                similarity_threshold = 0.74
                
                similarities = self.emb_handler.compute_similarity_matrix(mem_vecs, query_vec)
                best_idx = np.argmax(similarities)
                max_sim = similarities[best_idx]

                if max_sim >= similarity_threshold:
                    target_id = mem_ids[best_idx]
                    target_summary = mem_summaries[best_idx]
                    
                    cursor.execute("""
                        UPDATE memories 
                        SET status='archived', priority=0, last_used_at=? 
                        WHERE id=?
                    """, (datetime.now(), target_id))
                    
                    self.db.commit()
                    self._invalidate_cache()
                    logger.info(f"[MEMORY-FORGET] Forgot: {target_summary[:50]} (Sim: {max_sim:.2f})")
                    
                    return target_summary
                
                return None
            except Exception as e:
                logger.error(f"[MEMORY-FORGET] {e}")
                return None

    def deduplicate_existing_memories(self) -> int:
        """Vectorized deduplication using batched matrix similarity."""
        BATCH_SIZE = 200
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                cursor.execute("""
                    SELECT id, embedding FROM memories 
                    WHERE status='active' AND embedding IS NOT NULL
                    ORDER BY created_at ASC
                """)
                
                rows = cursor.fetchall()
                if len(rows) < 2:
                    return 0
                
                # Parse all embeddings into numpy arrays
                ids = []
                vecs = []
                for r_id, r_emb in rows:
                    vec = self.emb_handler.parse(r_emb)
                    if vec is not None:
                        ids.append(r_id)
                        vecs.append(vec)

                if len(vecs) < 2:
                    return 0

                to_remove = set()

                # Process in batches to limit memory usage
                for batch_start in range(0, len(vecs), BATCH_SIZE):
                    batch_end = min(batch_start + BATCH_SIZE, len(vecs))
                    batch_matrix = np.stack(vecs[batch_start:batch_end])

                    # Compare within this batch
                    sim_matrix = batch_matrix @ batch_matrix.T
                    # Only look at upper triangle (avoid self-comparison and double-counting)
                    dup_pairs = np.argwhere(
                        np.triu(sim_matrix, k=1) > self.similarity_threshold
                    )
                    for i_local, j_local in dup_pairs:
                        j_global = batch_start + int(j_local)
                        if ids[j_global] not in to_remove:
                            to_remove.add(ids[j_global])

                    # Compare this batch against subsequent batches
                    if batch_end < len(vecs):
                        rest_matrix = np.stack(vecs[batch_end:])
                        cross_sim = batch_matrix @ rest_matrix.T
                        cross_pairs = np.argwhere(cross_sim > self.similarity_threshold)
                        for i_local, j_local in cross_pairs:
                            i_global = batch_start + int(i_local)
                            j_global = batch_end + int(j_local)
                            # Keep the older one (lower index), archive the newer
                            if ids[i_global] not in to_remove and ids[j_global] not in to_remove:
                                to_remove.add(ids[j_global])

                if to_remove:
                    placeholders = ','.join(['?'] * len(to_remove))
                    cursor.execute(
                        f"UPDATE memories SET status='archived' WHERE id IN ({placeholders})",
                        list(to_remove)
                    )
                    self.db.commit()
                    self._invalidate_cache()
                    logger.info(f"[DEDUP] Removed {len(to_remove)} duplicates (vectorized)")
                    return len(to_remove)
                
                return 0
            except Exception as e:
                logger.error(f"[DEDUP] {e}")
                return 0


    def apply_decay_rules(self):
        IMMORTAL_TYPES = {
            MemoryType.FACT.value,
            MemoryType.PREFERENCE.value,
            MemoryType.DECISION.value,
            MemoryType.BOUNDARY.value,
        }
        with self._lock:
            now = datetime.now()
            cutoff_mood = now - timedelta(days=DECAY_DAYS_MOOD)
            cutoff_emotion = now - timedelta(days=DECAY_DAYS_EMOTION)
            cutoff_general = now - timedelta(days=DECAY_DAYS_GENERAL)

            immortal_placeholders = ','.join(['?'] * len(IMMORTAL_TYPES))
            query = f"""
                UPDATE memories SET status='archived'
                WHERE status='active'
                AND memory_type NOT IN ({immortal_placeholders})
                AND (
                    (memory_type=? AND last_used_at < ?) OR
                    (memory_type=? AND last_used_at < ?) OR
                    (memory_type NOT IN (?, ?) AND last_used_at < ?)
                )
            """
            params = tuple(list(IMMORTAL_TYPES) + [
                MemoryType.MOOD_STATE.value, cutoff_mood,
                MemoryType.EMOTION.value, cutoff_emotion,
                MemoryType.MOOD_STATE.value, MemoryType.EMOTION.value,
                cutoff_general
            ])

            max_retries = 3
            for attempt in range(1, max_retries + 1):
                try:
                    # Route write through DBConnection.execute_update so global write lock is used.
                    affected = self.db.execute_update(query, params)
                    if affected > 0:
                        logger.info(f"[DECAY] Archived {affected} memories")
                        self._invalidate_cache()
                    return
                except sqlite3.Error as e:
                    message = str(e).lower()
                    if "database is locked" in message and attempt < max_retries:
                        backoff = 0.2 * attempt
                        logger.warning(
                            f"[DECAY] database is locked (attempt {attempt}/{max_retries}), retrying in {backoff:.1f}s"
                        )
                        time.sleep(backoff)
                        continue
                    logger.error(f"[DECAY] {e}")
                    return
                except Exception as e:
                    logger.error(f"[DECAY] {e}")
                    return

    def optimize_memories(self, target_count: int = 500):
        with self._lock:
            try:
                cursor = self.db.get_cursor()
                
                cursor.execute(
                    "SELECT COUNT(*) FROM memories WHERE status='active'"
                )
                count = cursor.fetchone()[0]
                
                if count <= target_count:
                    return
                
                excess = count - target_count
                cursor.execute("""
                    UPDATE memories SET status='archived' WHERE id IN (
                        SELECT id FROM memories WHERE status='active'
                        ORDER BY (priority * 0.3 + use_count * 0.1 + 
                                 (julianday('now') - julianday(last_used_at)) * -0.01) ASC
                        LIMIT ?
                    )
                """, (excess,))
                
                if cursor.rowcount > 0:
                    self.db.commit()
                    self._invalidate_cache()
                    logger.info(f"[OPTIMIZE] Archived {cursor.rowcount} memories")
            except Exception as e:
                logger.error(f"[OPTIMIZE] {e}")

    def add_memory_batch(self, memories: List[Dict]):
        if not memories:
            return
            
        try:
            cursor = self.db.get_cursor()
            
            cursor.execute("SELECT lower(summary) FROM memories WHERE status='active'")
            existing_summaries = {row[0] for row in cursor.fetchall()}
            
            insert_data = []
            now = datetime.now()
            
            for mem in memories:
                summary = mem.get("summary", "").strip()
                if not summary or summary.lower() in existing_summaries:
                    continue
                
                existing_summaries.add(summary.lower())
                
                mem_id = str(uuid.uuid4())
                embedding_blob = np.array(mem["embedding"], dtype=np.float32).tobytes()
                
                insert_data.append(
                    (
                        mem_id,
                        summary,
                        mem["m_type"],
                        self._normalize_embedding_namespace(mem.get("embedding_namespace")),
                        mem["priority"],
                        now,
                        0,
                        'active',
                        embedding_blob,
                        now,
                    )
                )
                
            if insert_data:
                cursor.executemany("""
                    INSERT INTO memories (id, summary, memory_type, embedding_namespace, priority, 
                                        last_used_at, use_count, status, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, insert_data)
                self.db.commit()
                self._invalidate_cache()
                
            logger.info(f"[BATCH-INSERT] Stored {len(insert_data)} / {len(memories)} memories")
        except Exception as e:
            self.db.rollback()
            logger.error(f"[BATCH-INSERT] Failed: {e}")


    def get_memory_stats(self) -> Dict:
        now = time.monotonic()
        if "stats" in self._stats_cache and (now - self._stats_cache_ts) < self._stats_cache_ttl:
            return self._stats_cache["stats"]
        
        try:
            cursor = self.db.get_cursor()
            cursor.execute("""
                SELECT 
                    COUNT(*) as total,
                    SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active,
                    SUM(CASE WHEN status='archived' THEN 1 ELSE 0 END) as archived,
                    AVG(CASE WHEN status='active' THEN priority ELSE NULL END) as avg_priority,
                    MAX(last_used_at) as last_activity
                FROM memories
            """)
            
            row = cursor.fetchone()
            stats = {
                "total": row[0] or 0,
                "active": row[1] or 0,
                "archived": row[2] or 0,
                "avg_priority": round(row[3], 2) if row[3] else 0.0,
                "last_activity": row[4]
            }
            
            self._stats_cache["stats"] = stats
            self._stats_cache_ts = now
            return stats
        except Exception as e:
            logger.error(f"[STATS] {e}")
            return {"total": 0, "active": 0, "archived": 0, "avg_priority": 0.0}

    def get_top_memories(self, limit: int = 15) -> List[Dict]:
        """
        Return top-priority active memories ordered by priority DESC, created_at DESC.
        Each dict contains: summary, type, priority.
        Used by ProactiveEngine to populate the LTM section without bypassing DB abstraction.
        """
        try:
            cursor = self.db.get_cursor()
            cursor.execute(
                """SELECT summary, memory_type, priority, embedding_namespace FROM memories
                   WHERE status='active'
                   AND embedding_namespace = 'memory'
                   ORDER BY priority DESC, created_at DESC
                   LIMIT ?""",
                (limit,),
            )
            rows = cursor.fetchall()
            return [
                {"summary": row[0], "type": row[1], "priority": row[2], "embedding_namespace": row[3]}
                for row in rows
            ]
        except Exception as e:
            logger.error(f"[TOP-MEMORIES] Failed to fetch: {e}")
            return []

    def _invalidate_cache(self):
        self._stats_cache.pop("stats", None)
        self._stats_cache_ts = 0.0
