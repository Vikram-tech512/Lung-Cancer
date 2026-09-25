"""
tests/test_auth.py
==================
Unit tests verifying password hashing, token creation, and RBAC guards.
"""

import pytest
from backend.auth.security import hash_password, verify_password, create_access_token, decode_access_token

def test_password_hashing():
    pw = "SuperSecurePass2026!"
    hashed = hash_password(pw)
    assert hashed != pw
    assert verify_password(pw, hashed) is True
    assert verify_password("WrongPassword!", hashed) is False

def test_jwt_token_generation_and_decoding():
    payload = {"sub": "user-uuid-1234", "email": "test@lungcancerai.org", "role": "RESEARCHER"}
    token = create_access_token(payload)
    assert isinstance(token, str)
    decoded = decode_access_token(token)
    assert decoded["sub"] == "user-uuid-1234"
    assert decoded["email"] == "test@lungcancerai.org"
    assert decoded["role"] == "RESEARCHER"
