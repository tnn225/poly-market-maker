import sqlite3

class KeyValueStore:
    """Simple SQLite key-value store."""
    def __init__(self, db_path: str = "./data/kv_store.sqlite"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the key-value table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS kv_store (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def set(self, key: str, value: str):
        """Set a key-value pair."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO kv_store (key, value)
            VALUES (?, ?)
        """, (key, value))
        conn.commit()
        conn.close()
    
    def get(self, key: str) -> str:
        """Get a value by key. Returns None if key doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM kv_store WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None
    
    def delete(self, key: str):
        """Delete a key-value pair."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM kv_store WHERE key = ?", (key,))
        conn.commit()
        conn.close()
    
    def exists(self, key: str) -> bool:
        """Check if a key exists."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM kv_store WHERE key = ?", (key,))
        result = cursor.fetchone()
        conn.close()
        return result is not None
    
    def get_all(self) -> dict:
        """Get all key-value pairs as a dictionary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM kv_store")
        results = cursor.fetchall()
        conn.close()
        return dict(results)

