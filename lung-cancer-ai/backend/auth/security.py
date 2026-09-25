"""
backend/auth/security.py
========================
Cryptographic authentication, password hashing, JWT creation & verification,
and Role-Based Access Control (RBAC) security guards.
Supports PyJWT with zero-dependency HMAC fallback for host bootstrap.
"""

from __future__ import annotations
import hmac
import hashlib
import os
import datetime
import json
import base64
from typing import Optional, Dict, Any, List
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from backend.core.config import settings
from backend.database.database import get_db
from backend.database.models import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def hash_password(password: str, salt: Optional[str] = None) -> str:
    """Secure PBKDF2-HMAC password hashing."""
    if not salt:
        salt = os.urandom(16).hex()
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100000)
    return f"{salt}${key.hex()}"

def verify_password(plain_password: str, stored_hash: str) -> bool:
    try:
        salt, expected_hex = stored_hash.split("$")
        key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt.encode("utf-8"), 100000)
        return hmac.compare_digest(key.hex(), expected_hex)
    except Exception:
        return False

def _b64_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode('utf-8').rstrip('=')

def _b64_decode(data: str) -> bytes:
    padding = 4 - (len(data) % 4)
    if padding != 4:
        data += '=' * padding
    return base64.urlsafe_b64decode(data)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[datetime.timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": int(expire.timestamp())})

    try:
        import jwt
        return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    except ImportError:
        # Standard HMAC-SHA256 JWT generation
        header = json.dumps({"alg": "HS256", "typ": "JWT"}).encode("utf-8")
        payload = json.dumps(to_encode).encode("utf-8")
        segments = f"{_b64_encode(header)}.{_b64_encode(payload)}"
        signature = hmac.new(settings.SECRET_KEY.encode('utf-8'), segments.encode('utf-8'), hashlib.sha256).digest()
        return f"{segments}.{_b64_encode(signature)}"

def decode_access_token(token: str) -> Dict[str, Any]:
    try:
        import jwt
        return jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
    except ImportError:
        parts = token.split('.')
        if len(parts) != 3:
            raise ValueError("Invalid JWT format")
        segments = f"{parts[0]}.{parts[1]}"
        expected_sig = _b64_encode(hmac.new(settings.SECRET_KEY.encode('utf-8'), segments.encode('utf-8'), hashlib.sha256).digest())
        if not hmac.compare_digest(parts[2], expected_sig):
            raise ValueError("Invalid token signature")
        payload_data = json.loads(_b64_decode(parts[1]).decode('utf-8'))
        if "exp" in payload_data and payload_data["exp"] < datetime.datetime.utcnow().timestamp():
            raise ValueError("Token expired")
        return payload_data

def get_current_user(token: str = Depends(oauth2_scheme), db=Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = decode_access_token(token)
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except Exception:
        raise credentials_exception
    
    user = db.query(User).filter(User.id == user_id).first()
    if user is None or not user.is_active:
        raise credentials_exception
    return user

def require_role(allowed_roles: List[str]):
    def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {allowed_roles}"
            )
        return current_user
    return role_checker
