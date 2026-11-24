"""Engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from data.database.models import Base


@lru_cache
def get_engine(db_url: str) -> Engine:
    return create_engine(db_url, future=True)


def init_db(db_url: str) -> None:
    """Create any missing tables. Safe to run more than once."""
    Base.metadata.create_all(get_engine(db_url))


@contextmanager
def session_scope(db_url: str) -> Iterator[Session]:
    session = sessionmaker(bind=get_engine(db_url), expire_on_commit=False)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
