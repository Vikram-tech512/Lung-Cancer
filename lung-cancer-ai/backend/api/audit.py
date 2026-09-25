"""
backend/api/audit.py
====================
REST endpoints for system audit logging and compliance tracking.
"""
from __future__ import annotations

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import AuditLog

router = APIRouter(prefix="/audit", tags=["Audit"])


@router.get("")
def get_audit_trail(
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    resource_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Retrieves paginated immutable audit log events."""
    query = db.query(AuditLog)
    if resource_type:
        query = query.filter(AuditLog.resource_type == resource_type)

    total = query.count()
    events = query.order_by(AuditLog.timestamp.desc()).offset(offset).limit(limit).all()

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "events": [
            {
                "id": ev.id,
                "user_id": ev.user_id,
                "action": ev.action,
                "resource_type": ev.resource_type,
                "resource_id": ev.resource_id,
                "ip_address": ev.ip_address,
                "timestamp": ev.timestamp.isoformat() if ev.timestamp else None
            }
            for ev in events
        ]
    }
