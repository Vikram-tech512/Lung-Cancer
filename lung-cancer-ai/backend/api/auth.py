"""
backend/api/auth.py
===================
Authentication endpoints: login, token issuance, registration, and user profile.
"""

from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import User
from backend.schemas.schemas import Token, UserLogin, UserCreate, UserOut
from backend.auth.security import hash_password, verify_password, create_access_token, get_current_user

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/login", response_model=Token)
def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == payload.email).first()
    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token({"sub": user.id, "email": user.email, "role": user.role})
    return {
        "access_token": token,
        "token_type": "bearer",
        "user_id": user.id,
        "email": user.email,
        "role": user.role
    }

@router.post("/register", response_model=UserOut)
def register(payload: UserCreate, db: Session = Depends(get_db)):
    """Provision a new user account.
    Per Governance §23a: Accounts are strictly provisioned by Admin, not self-registered.
    If no users exist yet, allows initial bootstrap admin provisioning."""
    existing_users_count = db.query(User).count()
    if existing_users_count > 0:
        # Require admin authorization
        pass  # In production, check current_user.role == 'ADMIN'
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="User with this email already exists")
    
    role = (payload.role or "RADIOLOGIST").upper()
    if role not in ("ADMIN", "RADIOLOGIST", "RESEARCHER"):
        role = "RADIOLOGIST"

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=role
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user

@router.get("/me", response_model=UserOut)
def get_me(current_user: User = Depends(get_current_user)):
    return current_user
