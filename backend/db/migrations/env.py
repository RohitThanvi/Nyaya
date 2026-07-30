"""
Alembic environment script.

Reuses backend.config.settings.get_settings().db.sync_url instead of
alembic.ini's %(DB_USER)s-style interpolation, so migrations always
connect using the exact same host/user/password the running app itself
uses (env vars, .env file, or defaults) rather than needing a second,
separately-maintained source of DB connection info.
"""
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make `backend` importable when alembic is invoked from the repo root
# (alembic.ini also sets prepend_sys_path = . for the same reason, but
# this makes env.py robust even if invoked with a different cwd).
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from backend.config.settings import get_settings  # noqa: E402

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = None

_db_settings = get_settings().db
config.set_main_option("sqlalchemy.url", _db_settings.sync_url)


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
