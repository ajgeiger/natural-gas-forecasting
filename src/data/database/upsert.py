"""Insert-or-update so loaders can be rerun without duplicates."""

from __future__ import annotations

from typing import Any

from sqlalchemy.orm import Session

from data.database.models import Base, utc_now

# Keeps each statement under sqlite's bound parameter limit
CHUNK_SIZE = 500


def upsert(session: Session, model: type[Base], rows: list[dict[str, Any]]) -> int:
    if not rows:
        return 0

    dialect = session.get_bind().dialect.name
    if dialect == "sqlite":
        from sqlalchemy.dialects.sqlite import insert
    elif dialect == "postgresql":
        from sqlalchemy.dialects.postgresql import insert
    else:
        raise NotImplementedError(f"upsert not supported for {dialect}")

    table = model.__table__
    if "ingested_at" in table.columns:
        now = utc_now()
        rows = [{**row, "ingested_at": now} for row in rows]

    key_columns = [col.name for col in table.primary_key.columns]

    for start in range(0, len(rows), CHUNK_SIZE):
        stmt = insert(table).values(rows[start : start + CHUNK_SIZE])
        update_columns = {
            col.name: stmt.excluded[col.name]
            for col in table.columns
            if col.name not in key_columns and col.name in rows[0]
        }
        stmt = stmt.on_conflict_do_update(index_elements=key_columns, set_=update_columns)
        session.execute(stmt)

    return len(rows)
