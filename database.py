import sqlite3
import hashlib
from contextlib import contextmanager

# Database initialization
def init_db():
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

@contextmanager
def get_db():
    db = sqlite3.connect('internalgpt.db')
    db.row_factory = sqlite3.Row
    try:
        yield db
    finally:
        db.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# User operations
def create_user(email, password):
    hashed_password = hash_password(password)
    try:
        with get_db() as db:
            db.execute('INSERT INTO users (email, password) VALUES (?, ?)',
                      (email, hashed_password))
            db.commit()
            return True
    except sqlite3.IntegrityError:
        return False

def verify_user(email, password):
    hashed_password = hash_password(password)
    with get_db() as db:
        user = db.execute('SELECT * FROM users WHERE email = ? AND password = ?',
                         (email, hashed_password)).fetchone()
        return dict(user) if user else None

def update_password(email, new_password):
    hashed_password = hash_password(new_password)
    with get_db() as db:
        db.execute('UPDATE users SET password = ? WHERE email = ?',
                  (hashed_password, email))
        db.commit()

def get_user_by_email(email):
    with get_db() as db:
        user = db.execute('SELECT * FROM users WHERE email = ?', (email,)).fetchone()
        return dict(user) if user else None

# Initialize database when module is imported
init_db()
