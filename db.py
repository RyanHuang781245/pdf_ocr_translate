import sqlite3
from pathlib import Path

DB_PATH = Path("data/app.db")

SCHEMA = """
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  filename TEXT NOT NULL,
  upload_path TEXT NOT NULL,
  page_count INTEGER DEFAULT 0,
  created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS blocks (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  document_id INTEGER NOT NULL,
  page_no INTEGER NOT NULL,
  block_type TEXT NOT NULL,
  x0 REAL NOT NULL,
  y0 REAL NOT NULL,
  x1 REAL NOT NULL,
  y1 REAL NOT NULL,
  text_original TEXT NOT NULL,
  reading_order INTEGER DEFAULT 0,
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS translations (
  block_id INTEGER PRIMARY KEY,
  text_translated TEXT NOT NULL,
  updated_at TEXT DEFAULT (datetime('now')),
  FOREIGN KEY(block_id) REFERENCES blocks(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS annotations (
  document_id INTEGER NOT NULL,
  page_no INTEGER NOT NULL,
  canvas_json TEXT NOT NULL,
  updated_at TEXT DEFAULT (datetime('now')),
  PRIMARY KEY(document_id, page_no),
  FOREIGN KEY(document_id) REFERENCES documents(id) ON DELETE CASCADE
);
"""

def get_conn():
  DB_PATH.parent.mkdir(parents=True, exist_ok=True)
  conn = sqlite3.connect(DB_PATH)
  conn.row_factory = sqlite3.Row
  return conn

def init_db():
  conn = get_conn()
  conn.executescript(SCHEMA)
  conn.commit()
  conn.close()
