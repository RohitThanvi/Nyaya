"""
Auth route v2 — fixes refresh_token from query param to request body.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.config.settings import get_settings
from backend.db.session import get_db
from backend.models.domain import TokenPayload, TokenResponse, UserCreate, UserLogin

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

_cfg = get_settings().auth
_bearer = HTTPBearer(auto_error=False)


class RefreshRequest(BaseModel):
    refresh_token: str


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode(), hashed.encode())


def _create_token(data: dict, expires_delta: timedelta) -> str:
    payload = data.copy()
    payload["exp"] = datetime.now(timezone.utc) + expires_delta
    return jwt.encode(payload, _cfg.secret_key, algorithm=_cfg.algorithm)


def _create_access_token(user_id: str, role: str) -> str:
    return _create_token(
        {"sub": user_id, "role": role, "type": "access"},
        timedelta(minutes=_cfg.access_token_expire_minutes),
    )


def _create_refresh_token(user_id: str) -> str:
    return _create_token(
        {"sub": user_id, "type": "refresh"},
        timedelta(days=_cfg.refresh_token_expire_days),
    )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if not credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    try:
        payload = jwt.decode(credentials.credentials, _cfg.secret_key, algorithms=[_cfg.algorithm])
        user_id = payload.get("sub")
        if not user_id or payload.get("type") != "access":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    result = await db.execute(
        text("SELECT user_id, email, full_name, role, is_active FROM users WHERE user_id = :uid"),
        {"uid": user_id},
    )
    user = result.fetchone()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found or inactive")
    return {"user_id": str(user.user_id), "email": user.email, "full_name": user.full_name, "role": user.role}


import uuid as _uuid

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    existing = await db.execute(
        text("SELECT user_id FROM users WHERE email = :email"),
        {"email": user_data.email.lower().strip()},
    )
    if existing.fetchone():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email already registered")

    new_user_id = str(_uuid.uuid4())
    await db.execute(text("""
        INSERT INTO users (user_id, email, full_name, role, hashed_password, bar_enrollment, is_active)
        VALUES (:user_id, :email, :full_name, :role, :hashed_password, :bar_enrollment, true)
    """), {
        "user_id": new_user_id,
        "email": user_data.email.lower().strip(),
        "full_name": user_data.full_name.strip(),
        "role": user_data.role.value,
        "hashed_password": _hash_password(user_data.password),
        "bar_enrollment": user_data.bar_enrollment,
    })
    await db.commit()

    return TokenResponse(
        access_token=_create_access_token(new_user_id, user_data.role.value),
        refresh_token=_create_refresh_token(new_user_id),
        expires_in=_cfg.access_token_expire_minutes * 60,
    )


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    if not credentials.email or not credentials.password:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Email and password are required")
    result = await db.execute(
        text("SELECT user_id, hashed_password, role, is_active FROM users WHERE email = :email"),
        {"email": credentials.email.lower().strip()},
    )
    user = result.fetchone()
    if not user or not _verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Account disabled")

    user_id = str(user.user_id)
    await db.execute(
        text("UPDATE users SET last_login = NOW() WHERE user_id = :uid"),
        {"uid": user_id},
    )
    await db.commit()

    return TokenResponse(
        access_token=_create_access_token(user_id, user.role),
        refresh_token=_create_refresh_token(user_id),
        expires_in=_cfg.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshRequest,          # FIX: was query param, now request body
    db: AsyncSession = Depends(get_db),
):
    """Exchange refresh token for new access + refresh token pair."""
    try:
        payload = jwt.decode(body.refresh_token, _cfg.secret_key, algorithms=[_cfg.algorithm])
        user_id = payload.get("sub")
        if not user_id or payload.get("type") != "refresh":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    result = await db.execute(
        text("SELECT role, is_active FROM users WHERE user_id = :uid"),
        {"uid": user_id},
    )
    user = result.fetchone()
    if not user or not user.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    return TokenResponse(
        access_token=_create_access_token(user_id, user.role),
        refresh_token=_create_refresh_token(user_id),
        expires_in=_cfg.access_token_expire_minutes * 60,
    )


@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user
