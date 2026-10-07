"""Lightweight schema migration for the SQLite file.

`Base.metadata.create_all` only creates missing *tables*; it never alters an existing
one. A database written by an older MeetQ therefore keeps the old column set and the
app fails at runtime (e.g. "no such column: sessions.live"). This module brings such a
file up to the current models without needing a migration framework.
"""

from __future__ import annotations

import logging

from sqlalchemy import Engine, Table

from meetq.db.models import Base

log = logging.getLogger(__name__)


def _literal(value) -> str | None:
    """Render a Python default as a SQL literal, or None when we cannot."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        return f"'{escaped}'"
    return None


def _add_column_sql(table: Table, column, dialect) -> str:
    type_sql = column.type.compile(dialect)
    default = getattr(column.default, "arg", None) if column.default is not None else None
    literal = _literal(default) if not callable(default) else None
    parts = [f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {type_sql}']
    if literal is not None:
        # SQLite requires a default before it will add a NOT NULL column to existing rows
        if not column.nullable:
            parts.append("NOT NULL")
        parts.append(f"DEFAULT {literal}")
    return " ".join(parts)


def migrate(engine: Engine) -> list[str]:
    """Align the existing tables with the models. Returns the statements applied."""
    applied: list[str] = []
    dialect = engine.dialect
    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            rows = conn.exec_driver_sql(f'PRAGMA table_info("{table.name}")').fetchall()
            if not rows:
                continue  # create_all already made this table with the current shape
            existing = {row[1]: row for row in rows}

            for column in table.columns:
                if column.name in existing:
                    continue
                statement = _add_column_sql(table, column, dialect)
                conn.exec_driver_sql(statement)
                applied.append(statement)

            # A leftover NOT NULL column with no SQL default blocks every INSERT once
            # the model stops supplying it, so it has to go.
            for name, row in existing.items():
                if name in table.columns:
                    continue
                not_null, sql_default = row[3], row[4]
                if not_null and sql_default is None:
                    statement = f'ALTER TABLE "{table.name}" DROP COLUMN "{name}"'
                    conn.exec_driver_sql(statement)
                    applied.append(statement)

    for statement in applied:
        log.info("schema migration: %s", statement)
    return applied
