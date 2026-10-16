"""Database connectivity for the pgvector-backed movie embedding store."""

from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

_REQUIRED = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")


def _conninfo() -> dict[str, str]:
    missing = [k for k in _REQUIRED if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Copy .env.example to .env and fill it in."
        )
    return {
        "host": os.environ["DB_HOST"],
        "port": os.environ["DB_PORT"],
        "dbname": os.environ["DB_NAME"],
        "user": os.environ["DB_USER"],
        "password": os.environ["DB_PASSWORD"],
    }


@contextmanager
def get_connection() -> Iterator[psycopg.Connection]:
    """Yield a connection with the pgvector type adapter registered.

    register_vector() makes psycopg return `vector` columns as numpy arrays
    directly -- but only through a real cursor. Do NOT use pandas.read_sql
    with this connection: pandas bypasses the adapter and hands back raw
    text, which is what forced the old ast.literal_eval workaround.
    """
    conn = psycopg.connect(**_conninfo())
    try:
        register_vector(conn)
        yield conn
    finally:
        conn.close()
