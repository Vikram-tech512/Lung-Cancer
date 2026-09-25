"""
backend/api/settings.py
=======================
Application branding and custom domain configuration endpoints.
Allows changing APP_NAME, PUBLIC_DOMAIN, and viewing SSL/DNS setup.
"""

from __future__ import annotations
from fastapi import APIRouter
from backend.core.config import settings
from backend.schemas.schemas import DomainSettings

router = APIRouter(prefix="/settings", tags=["Settings & Custom Domain"])

# Dynamic in-memory configuration store for live updates
RUNTIME_SETTINGS = {
    "app_name": settings.APP_NAME,
    "public_domain": settings.PUBLIC_DOMAIN,
    "server_ip": "159.65.120.44",  # Example cloud VPS IP
    "ssl_status": "ACTIVE (Let's Encrypt TLS 1.3)"
}

@router.get("/domain", response_model=DomainSettings)
def get_domain_settings():
    return DomainSettings(
        app_name=RUNTIME_SETTINGS["app_name"],
        public_domain=RUNTIME_SETTINGS["public_domain"],
        server_ip=RUNTIME_SETTINGS["server_ip"],
        ssl_status=RUNTIME_SETTINGS["ssl_status"]
    )

@router.post("/domain", response_model=DomainSettings)
def update_domain_settings(payload: DomainSettings):
    RUNTIME_SETTINGS["app_name"] = payload.app_name
    RUNTIME_SETTINGS["public_domain"] = payload.public_domain
    RUNTIME_SETTINGS["server_ip"] = payload.server_ip
    return DomainSettings(**RUNTIME_SETTINGS)
