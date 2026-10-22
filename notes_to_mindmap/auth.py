"""Local user accounts: sign-up / log-in backed by SQLite (salted PBKDF2 hashes)."""
import hashlib, hmac, os, re, secrets, sqlite3, time
from contextlib import closing

ITERATIONS = 200_000
USER_RE = re.compile(r"^[A-Za-z0-9_]{3,30}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
BAD_LOGIN = "Invalid username or password."


def _db_path() -> str:
    return os.getenv("NOTES_DB", "users.db")


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(_db_path())
    c.execute("CREATE TABLE IF NOT EXISTS users("
              "username TEXT PRIMARY KEY COLLATE NOCASE, email TEXT, "
              "salt BLOB NOT NULL, pw_hash BLOB NOT NULL, created REAL NOT NULL)")
    return c


def _hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, ITERATIONS)


def register(username: str, email: str, password: str, confirm: str) -> tuple[bool, str]:
    """Create an account. Returns (ok, message)."""
    username, email = username.strip(), email.strip()
    if not USER_RE.match(username):
        return False, "Username must be 3-30 characters: letters, digits or underscore."
    if email and not EMAIL_RE.match(email):
        return False, "Please enter a valid email address (or leave it blank)."
    if len(password) < 8 or not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        return False, "Password needs at least 8 characters, with a letter and a digit."
    if password != confirm:
        return False, "Passwords do not match."
    salt = secrets.token_bytes(16)
    try:
        with closing(_conn()) as c, c:
            c.execute("INSERT INTO users VALUES(?,?,?,?,?)",
                      (username, email, salt, _hash(password, salt), time.time()))
    except sqlite3.IntegrityError:
        return False, "That username is already taken."
    return True, "Account created."


def authenticate(username: str, password: str) -> tuple[bool, str]:
    """Check credentials. Returns (True, canonical_username) or (False, error)."""
    with closing(_conn()) as c:
        row = c.execute("SELECT username, salt, pw_hash FROM users WHERE username=?",
                        (username.strip(),)).fetchone()
    if row is None:
        _hash(password, b"\0" * 16)          # same cost for unknown users
        return False, BAD_LOGIN
    name, salt, stored = row
    if hmac.compare_digest(_hash(password, salt), stored):
        return True, name
    return False, BAD_LOGIN
