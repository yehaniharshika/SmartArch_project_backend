from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import event
from sqlalchemy.engine import Engine

db = SQLAlchemy()

@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """
    Runs every time a new SQLite connection is opened.
    WAL mode = readers don't block writers, writers don't block readers.
    busy_timeout = if a write lock IS held briefly, retry instead of
    throwing 'database is locked' immediately.
    """
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=15000;")
    cursor.close()