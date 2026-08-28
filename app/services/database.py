import sqlite3
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CACHE_DIR = Path("cache")
DB_PATH = CACHE_DIR / "plans.db"

class DatabaseService:
    @staticmethod
    def init_db():
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS transition_plans (
                track_a TEXT,
                track_b TEXT,
                plan_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (track_a, track_b)
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("Database initialized at %s", DB_PATH)

    @staticmethod
    def get_plan(track_a: str, track_b: str) -> Optional[dict]:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute('SELECT plan_json FROM transition_plans WHERE track_a = ? AND track_b = ?', (track_a, track_b))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            logger.info(f"Cache HIT for transition: {track_a} -> {track_b}")
            return json.loads(row[0])
            
        logger.info(f"Cache MISS for transition: {track_a} -> {track_b}")
        return None

    @staticmethod
    def save_plan(track_a: str, track_b: str, plan_data: dict):
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        plan_json = json.dumps(plan_data)
        cursor.execute('''
            INSERT OR REPLACE INTO transition_plans (track_a, track_b, plan_json)
            VALUES (?, ?, ?)
        ''', (track_a, track_b, plan_json))
        conn.commit()
        conn.close()
        logger.info(f"Saved transition plan to cache: {track_a} -> {track_b}")

    @staticmethod
    def clear_cache():
        if DB_PATH.exists():
            DB_PATH.unlink()
        DatabaseService.init_db()
        logger.info("Database cache cleared.")
