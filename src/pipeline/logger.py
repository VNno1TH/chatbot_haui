"""
logger.py
Ghi log toàn bộ hội thoại, latency, intent để phục vụ Admin Dashboard.
Lưu vào SQLite (nhẹ, không cần cài thêm gì).

Đặt tại: src/pipeline/logger.py
"""

import sqlite3
import time
import json
import os
from pathlib import Path
from datetime import datetime
from threading import Lock

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "chat_logs.db"
_lock   = Lock()


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Tạo bảng nếu chưa có — gọi 1 lần lúc startup."""
    with _lock:
        conn = _get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_logs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT,
                turn         INTEGER,
                timestamp    TEXT,
                user_message TEXT,
                bot_answer   TEXT,
                intent       TEXT,
                method       TEXT,
                confidence   REAL,
                latency_ms   INTEGER,
                platform     TEXT DEFAULT 'web'
            );
            CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_logs(timestamp);
            CREATE INDEX IF NOT EXISTS idx_intent    ON chat_logs(intent);
        """)
        conn.commit()
        conn.close()


def log_chat(
    session_id  : str,
    turn        : int,
    user_message: str,
    bot_answer  : str,
    intent      : str,
    method      : str,
    confidence  : float,
    latency_ms  : int,
    platform    : str = "web",
):
    """Ghi 1 lượt hội thoại vào DB."""
    with _lock:
        conn = _get_conn()
        conn.execute(
            """INSERT INTO chat_logs
               (session_id, turn, timestamp, user_message, bot_answer,
                intent, method, confidence, latency_ms, platform)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                session_id,
                turn,
                datetime.now().isoformat(),
                user_message[:500],   # truncate
                bot_answer[:1000],
                intent,
                method,
                confidence,
                latency_ms,
                platform,
            ),
        )
        conn.commit()
        conn.close()


# ── Query functions cho Dashboard ────────────────────────────────────────────

def get_stats_overview() -> dict:
    """Tổng quan: tổng câu hỏi, latency trung bình, hôm nay."""
    conn = _get_conn()
    today = datetime.now().strftime("%Y-%m-%d")
    row = conn.execute("""
        SELECT
            COUNT(*)                          AS total,
            ROUND(AVG(latency_ms))            AS avg_latency,
            SUM(CASE WHEN timestamp LIKE ? THEN 1 ELSE 0 END) AS today,
            COUNT(DISTINCT session_id)        AS total_sessions
        FROM chat_logs
    """, (f"{today}%",)).fetchone()
    conn.close()
    return dict(row) if row else {}


def get_intent_stats(days: int = 7) -> list[dict]:
    """Phân bố intent trong N ngày gần nhất."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT intent, COUNT(*) AS count
        FROM chat_logs
        WHERE timestamp >= datetime('now', ?)
        GROUP BY intent
        ORDER BY count DESC
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_popular_questions(limit: int = 20) -> list[dict]:
    """Câu hỏi phổ biến nhất (group by nội dung tương tự)."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT user_message, COUNT(*) AS count,
               ROUND(AVG(latency_ms)) AS avg_latency,
               MAX(timestamp) AS last_seen
        FROM chat_logs
        GROUP BY user_message
        ORDER BY count DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latency_trend(days: int = 7) -> list[dict]:
    """Latency trung bình theo ngày."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            DATE(timestamp) AS date,
            ROUND(AVG(latency_ms))  AS avg_latency,
            ROUND(MIN(latency_ms))  AS min_latency,
            ROUND(MAX(latency_ms))  AS max_latency,
            COUNT(*)                AS count
        FROM chat_logs
        WHERE timestamp >= datetime('now', ?)
        GROUP BY DATE(timestamp)
        ORDER BY date
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_logs(limit: int = 50) -> list[dict]:
    """Log gần nhất để giám sát real-time."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT id, timestamp, session_id, turn,
               user_message, intent, method,
               confidence, latency_ms, platform
        FROM chat_logs
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_slow_queries(threshold_ms: int = 5000, limit: int = 20) -> list[dict]:
    """Các câu hỏi phản hồi chậm > threshold."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT timestamp, user_message, intent, latency_ms, method
        FROM chat_logs
        WHERE latency_ms > ?
        ORDER BY latency_ms DESC
        LIMIT ?
    """, (threshold_ms, limit)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_hourly_traffic(days: int = 1) -> list[dict]:
    """Traffic theo giờ trong N ngày."""
    conn = _get_conn()
    rows = conn.execute("""
        SELECT
            strftime('%H:00', timestamp) AS hour,
            COUNT(*) AS count
        FROM chat_logs
        WHERE timestamp >= datetime('now', ?)
        GROUP BY strftime('%H', timestamp)
        ORDER BY hour
    """, (f"-{days} days",)).fetchall()
    conn.close()
    return [dict(r) for r in rows]