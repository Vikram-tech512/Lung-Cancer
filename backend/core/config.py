"""
backend/core/config.py
======================
Central configuration management supporting custom domain, application name,
database credentials, redis broker, and security settings.
Includes zero-dependency fallback for host environments.
"""

from __future__ import annotations
import os

try:
    from pydantic_settings import BaseSettings
    class Settings(BaseSettings):
        APP_NAME: str = os.getenv("APP_NAME", "Lung Cancer AI")
        APP_SUBTITLE: str = "Multimodal CT Analysis, Segmentation & Explainable AI Platform"
        PUBLIC_DOMAIN: str = os.getenv("PUBLIC_DOMAIN", "lungcancerai.example.com")
        API_V1_STR: str = "/api"
        ENV: str = os.getenv("ENV", "production")
        SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-production-key-lung-cancer-ai-2026")
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
        DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/lung_cancer_ai.db")
        REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        STORAGE_ROOT: str = os.getenv("STORAGE_ROOT", "storage_bucket")
        MAX_UPLOAD_SIZE_MB: int = 500
        CLINICAL_DISCLAIMER: str = (
            "RESEARCH USE ONLY — NOT APPROVED FOR PRIMARY CLINICAL DIAGNOSIS. "
            "Algorithmic assessments represent decision-support outputs and require "
            "interpretation by qualified medical professionals."
        )
        class Config:
            case_sensitive = True
            env_file = ".env"
except ImportError:
    # Pure Python zero-dependency configuration fallback
    class Settings:
        APP_NAME: str = os.getenv("APP_NAME", "Lung Cancer AI")
        APP_SUBTITLE: str = "Multimodal CT Analysis, Segmentation & Explainable AI Platform"
        PUBLIC_DOMAIN: str = os.getenv("PUBLIC_DOMAIN", "lungcancerai.example.com")
        API_V1_STR: str = "/api"
        ENV: str = os.getenv("ENV", "production")
        SECRET_KEY: str = os.getenv("SECRET_KEY", "super-secret-production-key-lung-cancer-ai-2026")
        ALGORITHM: str = "HS256"
        ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
        DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///data/lung_cancer_ai.db")
        REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        STORAGE_ROOT: str = os.getenv("STORAGE_ROOT", "storage_bucket")
        MAX_UPLOAD_SIZE_MB: int = 500
        CLINICAL_DISCLAIMER: str = (
            "RESEARCH USE ONLY — NOT APPROVED FOR PRIMARY CLINICAL DIAGNOSIS. "
            "Algorithmic assessments represent decision-support outputs and require "
            "interpretation by qualified medical professionals."
        )

settings = Settings()
