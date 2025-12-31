import sqlite3
import json
from typing import Any, Optional
from contextlib import contextmanager
import requests

session = requests.Session()

class KeyValueStore:
    """
    A simple key-value store backed by SQLite.
    Supports storing any JSON-serializable values.
    """
    
    def __init__(self, db_path: str = "./data/cache.db"):
        """
        Initialize the key-value store.
        
        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the database and create the table if it doesn't exist."""
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS kv_store (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_updated_at ON kv_store(updated_at)
            """)
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        """Get a database connection with automatic commit/rollback."""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def set(self, key: str, value: Any) -> None:
        """
        Store a key-value pair.
        
        Args:
            key: The key to store
            value: The value to store (must be JSON-serializable)
        """
        try:
            json_value = json.dumps(value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Value must be JSON-serializable: {e}")
        
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO kv_store (key, value, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = CURRENT_TIMESTAMP
            """, (key, json_value))
            conn.commit()
    
    def get(self, key: str, default: Optional[Any] = None) -> Optional[Any]:
        """
        Retrieve a value by key.
        
        Args:
            key: The key to retrieve
            default: Default value to return if key doesn't exist
            
        Returns:
            The stored value, or default if key doesn't exist
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT value FROM kv_store WHERE key = ?",
                (key,)
            )
            row = cursor.fetchone()
            
            if row is None:
                return default
            
            try:
                return json.loads(row[0])
            except json.JSONDecodeError:
                # Fallback: return raw string if JSON parsing fails
                return row[0]
    
    def delete(self, key: str) -> bool:
        """
        Delete a key-value pair.
        
        Args:
            key: The key to delete
            
        Returns:
            True if key was deleted, False if it didn't exist
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM kv_store WHERE key = ?",
                (key,)
            )
            conn.commit()
            return cursor.rowcount > 0
    
    def exists(self, key: str) -> bool:
        """
        Check if a key exists in the store.
        
        Args:
            key: The key to check
            
        Returns:
            True if key exists, False otherwise
        """
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM kv_store WHERE key = ? LIMIT 1",
                (key,)
            )
            return cursor.fetchone() is not None
    
    def clear(self) -> None:
        """Clear all key-value pairs from the store."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM kv_store")
            conn.commit()
    
    def keys(self) -> list[str]:
        """
        Get all keys in the store.
        
        Returns:
            List of all keys
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT key FROM kv_store")
            return [row[0] for row in cursor.fetchall()]
    
    def size(self) -> int:
        """
        Get the number of key-value pairs in the store.
        
        Returns:
            Number of entries
        """
        with self._get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM kv_store")
            return cursor.fetchone()[0]
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator: key in store"""
        return self.exists(key)
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-style access: store[key]"""
        value = self.get(key)
        if value is None and not self.exists(key):
            raise KeyError(f"Key '{key}' not found")
        return value
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Support dictionary-style assignment: store[key] = value"""
        self.set(key, value)
    
    def __delitem__(self, key: str) -> None:
        """Support dictionary-style deletion: del store[key]"""
        if not self.delete(key):
            raise KeyError(f"Key '{key}' not found")

    def get_key(self, base_url: str, params: dict) -> str:
        """
        Generate a cache key from a base URL and parameters.
        Returns the full URL with query parameters properly encoded.
        
        Args:
            base_url: The base URL
            params: Dictionary of query parameters
            
        Returns:
            Full URL with query parameters
        """
        req = requests.Request('GET', base_url, params=params)
        prepared = req.prepare()
        return prepared.url
        
    def get_cached(self, base_url: str, params: dict) -> Any:
        """
        Fetch data from a URL with caching.
        If the URL is already cached, returns cached data.
        Otherwise, fetches from the URL and caches the result.
        
        Args:
            base_url: The base URL to fetch from
            params: Dictionary of query parameters
            
        Returns:
            The JSON response data (cached or fresh)
        """
        url = self.get_key(base_url, params)
        if self.exists(url):
            return self.get(url)
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        self.set(url, data)
        return data

