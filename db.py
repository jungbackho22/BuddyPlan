from __future__ import annotations
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import datetime

DB_PATH = Path(__file__).parent / "db" / "app.db"

def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=WAL;")
    return con

def init_db() -> None:
    con = connect()
    con.execute("""
    CREATE TABLE IF NOT EXISTS assessments (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      responses_json TEXT NOT NULL,
      factor_scores_json TEXT NOT NULL,
      top_factors_json TEXT NOT NULL,
      recommended_scenarios_json TEXT NOT NULL
    );
    """)
    con.execute("""
    CREATE TABLE IF NOT EXISTS scenario_attempts (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id TEXT NOT NULL,
      created_at TEXT NOT NULL,
      scenario_id TEXT NOT NULL,
      chosen_label TEXT NOT NULL,
      is_correct INTEGER NOT NULL
    );
    """)
    con.commit()
    con.close()

def now_iso() -> str:
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def insert_assessment(
    user_id: str,
    responses: Dict[str, int],
    factor_scores: List[Dict[str, Any]],
    top_factors: List[Dict[str, Any]],
    recommended_ids: List[str],
) -> int:
    con = connect()
    cur = con.cursor()
    cur.execute(
        """INSERT INTO assessments
        (user_id, created_at, responses_json, factor_scores_json, top_factors_json, recommended_scenarios_json)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            now_iso(),
            json.dumps(responses, ensure_ascii=False),
            json.dumps(factor_scores, ensure_ascii=False),
            json.dumps(top_factors, ensure_ascii=False),
            json.dumps(recommended_ids, ensure_ascii=False),
        ),
    )
    con.commit()
    rid = int(cur.lastrowid)
    con.close()
    return rid

def latest_assessment(user_id: str) -> Optional[Dict[str, Any]]:
    con = connect()
    row = con.execute(
        """SELECT id, created_at, factor_scores_json, top_factors_json, recommended_scenarios_json
             FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT 1""",
        (user_id,),
    ).fetchone()
    con.close()
    if not row:
        return None
    return {
        "id": row[0],
        "created_at": row[1],
        "factor_scores": json.loads(row[2]),
        "top_factors": json.loads(row[3]),
        "recommended_ids": json.loads(row[4]),
    }

def list_assessments(user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
    con = connect()
    rows = con.execute(
        """SELECT id, created_at, top_factors_json
             FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        out.append({"id": r[0], "created_at": r[1], "top_factors": json.loads(r[2])})
    return out

def insert_attempt(user_id: str, scenario_id: str, chosen_label: str, is_correct: bool) -> None:
    con = connect()
    con.execute(
        """INSERT INTO scenario_attempts
             (user_id, created_at, scenario_id, chosen_label, is_correct)
             VALUES (?, ?, ?, ?, ?)""",
        (user_id, now_iso(), scenario_id, chosen_label, 1 if is_correct else 0),
    )
    con.commit()
    con.close()
