"""
SQLite-backed user authentication database.

Replaces the hardcoded USERS dict with a proper database and
bcrypt-hashed passwords via werkzeug.security.
"""

import os
import sqlite3
import logging
from werkzeug.security import generate_password_hash, check_password_hash

logger = logging.getLogger("AuthDB")

_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "users.db"
)
_DB_PATH = os.path.normpath(_DB_PATH)


def _get_conn():
    """Return a new SQLite connection with row factory."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the users table if it doesn't exist and seed the default admin."""
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT    UNIQUE NOT NULL,
                password TEXT    NOT NULL,
                role     TEXT    NOT NULL DEFAULT 'operator'
            )
        """)
        conn.commit()

        # Seed default admin if the table is empty
        row = conn.execute("SELECT COUNT(*) AS cnt FROM users").fetchone()
        if row["cnt"] == 0:
            create_user("admin", "admin123", role="admin", _conn=conn)
            create_user("operator", "operator123", role="operator", _conn=conn)
            logger.info("Seeded default users: admin, operator")
    finally:
        conn.close()


def create_user(username, password, role="operator", _conn=None):
    """Insert a new user with a hashed password."""
    pw_hash = generate_password_hash(password, method="pbkdf2:sha256")
    close = _conn is None
    conn = _conn or _get_conn()
    try:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, pw_hash, role),
        )
        conn.commit()
        logger.info(f"Created user '{username}' with role '{role}'")
    finally:
        if close:
            conn.close()


def get_user_by_username(username):
    """Return a user dict or None."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, password, role FROM users WHERE username = ?",
            (username,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def get_user_by_id(user_id):
    """Return a user dict or None (used by Flask-Login user_loader)."""
    conn = _get_conn()
    try:
        row = conn.execute(
            "SELECT id, username, password, role FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def verify_password(username, password):
    """Check a plain-text password against the stored hash. Returns True/False."""
    user = get_user_by_username(username)
    if user is None:
        return False
    return check_password_hash(user["password"], password)
