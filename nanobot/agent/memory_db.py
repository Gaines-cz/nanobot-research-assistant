"""
SQLite 操作层 - 记忆系统数据库接口

提供基础的 SQLite CRUD 操作，支持软删除、搜索和阅读统计更新。
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class MemoryDatabase:
    """SQLite-backed memory database."""

    def __init__(self, db_path: Path):
        """
        初始化 SQLite 连接，创建表结构。

        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db_directory()
        self._connect()
        self._create_tables()

    def _ensure_db_directory(self) -> None:
        """确保数据库目录存在"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> None:
        """建立数据库连接"""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        # 启用 WAL 模式，提高并发性能
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def _create_tables(self) -> None:
        """创建表结构"""
        sql = """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            detail TEXT NOT NULL,
            at_time INTEGER NOT NULL,
            read_times INTEGER DEFAULT 0,
            last_read_time INTEGER,
            created_at INTEGER NOT NULL,
            updated_at INTEGER NOT NULL,
            deleted_at INTEGER
        );
        """
        self._conn.execute(sql)

        # 创建索引
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories(type) WHERE deleted_at IS NULL"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_at_time ON memories(at_time) WHERE deleted_at IS NULL"
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_memories_last_read ON memories(last_read_time) WHERE deleted_at IS NULL"
        )

        self._conn.commit()
        logger.info(f"MemoryDatabase initialized at {self.db_path}")

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        """将 sqlite3.Row 转换为字典"""
        return dict(row)

    def _now(self) -> int:
        """获取当前 Unix 时间戳（秒）"""
        return int(time.time())

    def insert(self, type: str, detail: str, at_time: int, read_times: int = 0) -> int:
        """
        新增记忆，返回生成的 id。

        Args:
            type: 记忆类型 ('HISTORY' | 'KNOWLEDGE' | 'DECISIONS' | 'PROJECTS')
            detail: 记忆内容（Markdown 格式）
            at_time: 记忆关联的时间戳（Unix 秒）
            read_times: 初始阅读次数，默认为 0

        Returns:
            生成的 INTEGER id (AUTOINCREMENT)

        Raises:
            sqlite3.Error: 数据库操作失败
        """
        now = self._now()

        # last_read_time 默认为 created_at
        last_read_time = now if read_times == 0 else now

        sql = """
        INSERT INTO memories (type, detail, at_time, read_times, last_read_time, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        try:
            cursor = self._conn.execute(
                sql,
                (type, detail, at_time, read_times, last_read_time, now, now)
            )
            self._conn.commit()
            memory_id = cursor.lastrowid
            logger.debug(f"Inserted memory: id={memory_id}, type={type}")
            return memory_id
        except sqlite3.Error as e:
            logger.error(f"Failed to insert memory: {e}")
            raise

    def soft_delete(self, id: int) -> bool:
        """
        软删除记忆（设置 deleted_at 时间戳）。

        Args:
            id: 记忆 ID

        Returns:
            True 表示成功，False 表示未找到或已删除
        """
        now = self._now()
        sql = "UPDATE memories SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL"
        cursor = self._conn.execute(sql, (now, id))
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Soft deleted memory: id={id}")
        return deleted

    def hard_delete(self, id: int) -> bool:
        """
        硬删除记忆（物理删除）。

        Args:
            id: 记忆 ID

        Returns:
            True 表示成功，False 表示未找到
        """
        sql = "DELETE FROM memories WHERE id = ?"
        cursor = self._conn.execute(sql, (id,))
        self._conn.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.debug(f"Hard deleted memory: id={id}")
        return deleted

    def update(self, id: int, detail: str, at_time: int) -> bool:
        """
        更新记忆的 detail 和 at_time。

        Args:
            id: 记忆 ID
            detail: 新的记忆内容
            at_time: 新的关联时间戳

        Returns:
            True 表示成功，False 表示未找到或已删除
        """
        now = self._now()
        sql = """
        UPDATE memories
        SET detail = ?, at_time = ?, updated_at = ?
        WHERE id = ? AND deleted_at IS NULL
        """
        cursor = self._conn.execute(sql, (detail, at_time, now, id))
        self._conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.debug(f"Updated memory: id={id}")
        return updated

    def get_by_id(self, id: int) -> Optional[dict]:
        """
        按 ID 获取记忆（不包含软删除的记录）。

        Args:
            id: 记忆 ID

        Returns:
            记忆字典，或 None（不存在或已删除）
        """
        sql = "SELECT * FROM memories WHERE id = ? AND deleted_at IS NULL"
        cursor = self._conn.execute(sql, (id,))
        row = cursor.fetchone()
        return self._row_to_dict(row) if row else None

    def search(self, query: str, type: Optional[str] = None, top_k: int = 10) -> list[dict]:
        """
        按 type 搜索记忆（不包含软删除的记录）。
        使用 LIKE 进行全文搜索。

        Args:
            query: 搜索关键词
            type: 可选的记忆类型过滤
            top_k: 返回的最大结果数

        Returns:
            匹配的记忆列表（按 at_time 降序排列）
        """
        if type:
            sql = """
            SELECT * FROM memories
            WHERE type = ? AND deleted_at IS NULL AND detail LIKE ?
            ORDER BY at_time DESC
            LIMIT ?
            """
            pattern = f"%{query}%"
            cursor = self._conn.execute(sql, (type, pattern, top_k))
        else:
            sql = """
            SELECT * FROM memories
            WHERE deleted_at IS NULL AND detail LIKE ?
            ORDER BY at_time DESC
            LIMIT ?
            """
            pattern = f"%{query}%"
            cursor = self._conn.execute(sql, (pattern, top_k))

        rows = cursor.fetchall()
        return [self._row_to_dict(row) for row in rows]

    def update_read_stats(self, id: int) -> bool:
        """
        更新记忆的阅读统计（read_times++, last_read_time=now）。

        Args:
            id: 记忆 ID

        Returns:
            True 表示成功，False 表示未找到或已删除
        """
        now = self._now()
        sql = """
        UPDATE memories
        SET read_times = read_times + 1, last_read_time = ?
        WHERE id = ? AND deleted_at IS NULL
        """
        cursor = self._conn.execute(sql, (now, id))
        self._conn.commit()
        updated = cursor.rowcount > 0
        if updated:
            logger.debug(f"Updated read stats for memory: id={id}")
        return updated

    @property
    def connection(self) -> sqlite3.Connection:
        """Public access to database connection."""
        return self._conn

    def close(self) -> None:
        """关闭数据库连接"""
        if self._conn:
            self._conn.close()
            self._conn = None
            logger.debug("MemoryDatabase connection closed")

    def __enter__(self) -> "MemoryDatabase":
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """上下文管理器出口"""
        self.close()

    def __del__(self) -> None:
        """析构时确保关闭连接"""
        self.close()
