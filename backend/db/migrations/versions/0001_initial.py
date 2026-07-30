"""
Baseline marker — NOT the real initial schema.

This file used to independently define the whole schema via SQLAlchemy
op.create_table() calls, as an early draft that predates
backend/db/migrations/0001_initial.sql. It was never deleted, and it
diverged: e.g. its `chunks` table has no `law` column, and several other
tables differ (chat_sessions.law_context vs law_filter, audit_logs
.resource_type vs .resource, etc.).

The schema actually in use everywhere (docker-entrypoint-initdb.d on
first container start, and backend/main.py's startup SQL-glob runner on
every boot) is 0001_initial.sql. If this file's upgrade() ran for real,
it would either try to CREATE TABLE users/documents/chunks/... that
already exist (erroring out and blocking every later revision from
running), or — on a from-scratch DB with no docker init — silently
build the wrong, `law`-less schema instead.

Kept as a no-op so the Alembic chain (0002+ all descend from
"0001_initial") still resolves and `alembic upgrade head` can run
safely against a database whose baseline was created by the .sql file,
not by this one.
"""

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Real baseline schema lives in 0001_initial.sql — applied via
    # docker-entrypoint-initdb.d and backend/main.py's startup migration
    # runner, not via Alembic. Nothing to do here.
    pass


def downgrade() -> None:
    # Intentionally not reversible from here — this revision no longer
    # owns the schema it's named after.
    pass
