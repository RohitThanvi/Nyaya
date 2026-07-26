"""
Auth route v3 — consolidated onto the canonical auth implementation.

FIX: this file used to maintain its own bcrypt hashing, JWT encode/decode,
and get_current_user, completely separate from backend.api.dependencies.auth
(which every other route in the app actually depends on). Two problems:
1. This local _hash_password/_verify_password never truncated to bcrypt's
   72-byte hard limit — confirmed against the pinned bcrypt version that
   hashing a >72-byte password raises an unhandled ValueError, so any user
   with a longer passphrase got a raw 500 on /register and /login.
2. Two independent copies of security-critical logic is a real hazard: a
   fix applied to one (e.g. an inactive-user check, a lockout policy) has
   no guarantee of being applied to the other. Now uses the single
   canonical implementation everywhere.
"""
import logging
import uuid as _uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from jose import JWTError, jwt
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.auth import (
    create_access_token, create_refresh_token, get_current_user,
    hash_password, verify_password,
)
from backend.config.settings import get_settings
from backend.db.session import get_db
from backend.models.domain import TokenResponse, UserCreate, UserLogin, UserInDB

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger(__name__)

_cfg = get_settings().auth


class RefreshRequest(BaseModel):
    refresh_token: str


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
        "hashed_password": hash_password(user_data.password),
        "bar_enrollment": user_data.bar_enrollment,
    })
    await db.commit()

    return TokenResponse(
        access_token=create_access_token(new_user_id, user_data.role.value),
        refresh_token=create_refresh_token(new_user_id),
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
    if not user or not verify_password(credentials.password, user.hashed_password):
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
        access_token=create_access_token(user_id, user.role),
        refresh_token=create_refresh_token(user_id),
        expires_in=_cfg.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    body: RefreshRequest,          # FIX (kept from v2): was query param, now request body
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
        access_token=create_access_token(user_id, user.role),
        refresh_token=create_refresh_token(user_id),
        expires_in=_cfg.access_token_expire_minutes * 60,
    )


@router.get("/me")
async def get_me(current_user: UserInDB = Depends(get_current_user)):
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
    }
