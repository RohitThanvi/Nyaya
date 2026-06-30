"""Health check route."""
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from backend.db.session import get_db
from backend.config.settings import get_settings

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
async def health(db: AsyncSession = Depends(get_db)):
    cfg = get_settings()
    db_ok = False
    try:
        await db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        pass
    return {
        "status": "ok" if db_ok else "degraded",
        "version": cfg.app.app_version,
        "environment": cfg.app.environment,
        "db": "ok" if db_ok else "error",
    }
