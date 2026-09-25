"""
backend/database/database.py
============================
SQLAlchemy database engine and session factory with connection pooling
and resilient fallback for zero-dependency initialization.
"""

from __future__ import annotations
import os
import logging
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

from backend.core.config import settings

logger = logging.getLogger("backend.database")

Base = declarative_base()

def get_engine():
    db_url = settings.DATABASE_URL
    if db_url.startswith("sqlite"):
        # Resolve SQLite file relative to project root
        db_path = Path(__file__).parent.parent.parent / "data" / "lung_cancer_ai.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(f"sqlite:///{db_path.resolve().as_posix()}", connect_args={"check_same_thread": False})
    
    try:
        eng = create_engine(db_url, pool_size=10, max_overflow=20, pool_pre_ping=True)
        with eng.connect() as conn:
            pass
        return eng
    except Exception as e:
        logger.warning(f"Could not connect to PostgreSQL ({e}). Falling back to local SQLite database.")
        db_path = Path(__file__).parent.parent.parent / "data" / "lung_cancer_ai.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return create_engine(f"sqlite:///{db_path.resolve().as_posix()}", connect_args={"check_same_thread": False})

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def get_db():
    """Dependency injection helper for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
