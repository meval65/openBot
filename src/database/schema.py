import sqlite3
import logging

logger = logging.getLogger(__name__)

def create_tables(conn: sqlite3.Connection, lock):
    with lock:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                summary TEXT NOT NULL,
                content_text TEXT,
                group_id TEXT,
                memory_type TEXT NOT NULL,
                embedding_namespace TEXT NOT NULL DEFAULT 'memory',
                priority REAL DEFAULT 0.5,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 0,
                status TEXT DEFAULT 'active',
                embedding BLOB,
                CHECK(priority >= 0.0 AND priority <= 1.0),
                CHECK(status IN ('active', 'archived', 'deleted')),
                CHECK(embedding_namespace IN ('memory', 'document', 'image'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schedules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheduled_at TIMESTAMP NOT NULL,
                context TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending',
                recurrence TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                executed_at TIMESTAMP,
                execution_note TEXT,
                metadata TEXT,
                tags TEXT,
                CHECK(status IN ('pending', 'executed', 'cancelled', 'failed'))
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS memory_groups (
                id TEXT PRIMARY KEY,
                file_path TEXT,
                file_hash TEXT UNIQUE NOT NULL,
                media_type TEXT NOT NULL DEFAULT 'image',
                description TEXT,
                tile_count INTEGER DEFAULT 1,
                width INTEGER,
                height INTEGER,
                original_width INTEGER,
                original_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sticker_video_descriptions (
                file_hash TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS video_media_cache (
                original_hash TEXT PRIMARY KEY,
                optimized_path TEXT NOT NULL,
                optimized_mime TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS media_descriptions (
                media_hash TEXT PRIMARY KEY,
                media_kind TEXT NOT NULL,
                description TEXT NOT NULL,
                file_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 0,
                CHECK(media_kind IN ('image', 'video', 'sticker_video'))
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_image_sources (
                source_url TEXT PRIMARY KEY,
                media_hash TEXT NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS web_image_raw_hashes (
                raw_hash TEXT PRIMARY KEY,
                media_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                use_count INTEGER DEFAULT 0
            )
        """)

        conn.commit()


def create_indexes(conn: sqlite3.Connection, lock):
    with lock:
        cursor = conn.cursor()
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_core ON memories(status, memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_namespace ON memories(status, embedding_namespace)",
            "CREATE INDEX IF NOT EXISTS idx_memories_ranking ON memories(priority DESC, last_used_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_memories_embedding ON memories(embedding) WHERE embedding IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_schedules_pending ON schedules(status, scheduled_at)",
            "CREATE INDEX IF NOT EXISTS idx_schedules_lookup ON schedules(scheduled_at)",
            "CREATE INDEX IF NOT EXISTS idx_memories_group ON memories(group_id) WHERE group_id IS NOT NULL",
            "CREATE INDEX IF NOT EXISTS idx_memory_groups_hash ON memory_groups(file_hash)",
            "CREATE INDEX IF NOT EXISTS idx_sticker_video_last_used ON sticker_video_descriptions(last_used_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_video_media_last_used ON video_media_cache(last_used_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_media_desc_kind_used ON media_descriptions(media_kind, last_used_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_web_image_sources_hash ON web_image_sources(media_hash)",
            "CREATE INDEX IF NOT EXISTS idx_web_image_sources_used ON web_image_sources(last_used_at DESC)",
            "CREATE INDEX IF NOT EXISTS idx_web_image_raw_hashes_media ON web_image_raw_hashes(media_hash)",
        ]
        
        for idx_sql in indexes:
            try:
                cursor.execute(idx_sql)
            except sqlite3.Error as e:
                logger.warning(f"[DATABASE] Index creation skipped: {idx_sql[:60]}... ({e})")
        
        conn.commit()


def migrate_schema(conn: sqlite3.Connection, lock):
    with lock:
        cursor = conn.cursor()
        try:
            table_columns = {
                'schedules': [
                    ('recurrence', 'TEXT'),
                    ('executed_at', 'TIMESTAMP'),
                    ('execution_note', 'TEXT'),
                    ('metadata', 'TEXT'),
                    ('tags', 'TEXT')
                ],
                'memories': [
                    ('content_text', 'TEXT'),
                    ('group_id', 'TEXT'),
                    ('embedding_namespace', "TEXT NOT NULL DEFAULT 'memory'"),
                ],
                'media_descriptions': [
                    ('file_path', 'TEXT'),
                ],
                'web_image_sources': [
                    ('description', 'TEXT'),
                ],
                'web_image_raw_hashes': [],
            }
            
            for table, columns in table_columns.items():
                cursor.execute(f"PRAGMA table_info({table})")
                existing = {col[1] for col in cursor.fetchall()}
                
                for col_name, col_type in columns:
                    if col_name not in existing:
                        try:
                            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                        except sqlite3.OperationalError:
                            pass

            conn.commit()
        except Exception as e:
            logger.error(f"[DATABASE] Schema migration failed: {e}")
