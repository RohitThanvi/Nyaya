"""Auth routes — register, login, refresh."""
from datetime import datetime
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.dependencies.auth import (
    create_access_token, create_refresh_token, decode_token,
    get_current_active_user, hash_password, verify_password
)
from backend.db.session import get_db
from backend.models.domain import TokenResponse, UserCreate, UserInDB, UserRole

router = APIRouter()


@router.post("/register", response_model=TokenResponse, status_code=201)
async def register(body: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    existing = await db.execute(
        text("SELECT user_id FROM users WHERE email = :email"), {"email": body.email}
    )
    if existing.fetchone():
        raise HTTPException(400, "Email already registered")

    hashed = hash_password(body.password)
    result = await db.execute(
        text("""
            INSERT INTO users (user_id, email, full_name, hashed_password, role, bar_enrollment, is_active)
            VALUES (:user_id, :email, :name, :pwd, :role, :bar, :is_active)
            RETURNING user_id
        """),
        {
            "user_id": str(uuid.uuid4()),
            "email": body.email,
            "name": body.full_name,
            "pwd": hashed,
            "role": body.role.value,
            "bar": body.bar_enrollment,
            "is_active": True,
        },
    )
    user_id = str(result.scalar())
    await db.commit()

    access = create_access_token(user_id, body.role.value)
    refresh = create_refresh_token(user_id)
    from backend.config.settings import get_settings
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=get_settings().auth.access_token_expire_minutes * 60,
    )


@router.post("/login", response_model=TokenResponse)
async def login(body: UserCreate, db: AsyncSession = Depends(get_db)):
    """Authenticate user and return tokens."""
    result = await db.execute(
        text("SELECT * FROM users WHERE email = :email AND is_active = true"),
        {"email": body.email},
    )
    user = result.fetchone()
    if not user or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )
    # Update last login
    await db.execute(
        text("UPDATE users SET last_login = :ts WHERE user_id = :uid"),
        {"ts": datetime.utcnow(), "uid": str(user.user_id)},
    )
    await db.commit()

    access = create_access_token(str(user.user_id), user.role)
    refresh = create_refresh_token(str(user.user_id))
    from backend.config.settings import get_settings
    return TokenResponse(
        access_token=access,
        refresh_token=refresh,
        expires_in=get_settings().auth.access_token_expire_minutes * 60,
    )


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    """Refresh access token using refresh token."""
    payload = decode_token(refresh_token)
    result = await db.execute(
        text("SELECT role FROM users WHERE user_id = :uid AND is_active = true"),
        {"uid": payload.sub},
    )
    user = result.fetchone()
    if not user:
        raise HTTPException(401, "User not found or inactive")
    access = create_access_token(payload.sub, user.role)
    new_refresh = create_refresh_token(payload.sub)
    from backend.config.settings import get_settings
    return TokenResponse(
        access_token=access,
        refresh_token=new_refresh,
        expires_in=get_settings().auth.access_token_expire_minutes * 60,
    )


@router.get("/me")
async def get_me(current_user: UserInDB = Depends(get_current_active_user)):
    return {
        "user_id": current_user.user_id,
        "email": current_user.email,
        "full_name": current_user.full_name,
        "role": current_user.role.value,
    }
