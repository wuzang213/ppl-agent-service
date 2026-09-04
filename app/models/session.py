import sqlite3
import uuid
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from pydantic import BaseModel


# Pydantic 模型
class SessionCreate(BaseModel):
    """创建会话的请求模型"""
    user_id: str
    biz_type: str
    name: str


class SessionResponse(BaseModel):
    """会话响应模型"""
    thread_id: str
    user_id: str
    biz_type: str
    name: str
    created_at: str
    updated_at: str


# 数据库配置
# __file__ = app/models/session.py
# parent.parent.parent → agent-service 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "sessions.db"
# 旧路径
# DB_PATH = Path(__file__).parent.parent / "db/sessions.db"


def get_db_connection():
    """获取数据库连接"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """初始化数据库，创建 sessions 表"""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            thread_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            biz_type TEXT NOT NULL,
            name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_session(session: SessionCreate) -> SessionResponse:
    """创建新会话"""
    conn = get_db_connection()
    cursor = conn.cursor()

    thread_id = str(uuid.uuid4())
    now = datetime.now().isoformat()

    cursor.execute(
        """
        INSERT INTO sessions (thread_id, user_id, biz_type, name, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (thread_id, session.user_id, session.biz_type, session.name, now, now)
    )
    conn.commit()
    conn.close()

    return SessionResponse(
        thread_id=thread_id,
        user_id=session.user_id,
        biz_type=session.biz_type,
        name=session.name,
        created_at=now,
        updated_at=now
    )


def get_sessions(user_id: Optional[str] = None, biz_type: Optional[str] = None) -> List[SessionResponse]:
    """查询会话列表，支持按 user_id 和 biz_type 筛选"""
    conn = get_db_connection()
    cursor = conn.cursor()

    query = "SELECT thread_id, user_id, biz_type, name, created_at, updated_at FROM sessions"
    params = []

    if user_id or biz_type:
        conditions = []
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)
        if biz_type:
            conditions.append("biz_type = ?")
            params.append(biz_type)
        query += " WHERE " + " AND ".join(conditions)

    query += " ORDER BY updated_at DESC"

    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()

    return [
        SessionResponse(
            thread_id=row["thread_id"],
            user_id=row["user_id"],
            biz_type=row["biz_type"],
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"]
        )
        for row in rows
    ]


def delete_session(thread_id: str) -> bool:
    """删除会话"""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("DELETE FROM sessions WHERE thread_id = ?", (thread_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()

    return deleted


# 初始化数据库
init_db()