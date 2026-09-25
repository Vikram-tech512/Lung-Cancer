"""
backend/main.py
===============
Production FastAPI application for the Lung Cancer AI Research Platform.
Exposes REST APIs, WebSocket streaming, and interactive API documentation (/docs).
"""

from __future__ import annotations
import asyncio
import json
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from backend.core.config import settings
from backend.database.database import engine, Base
from backend.websocket.manager import ws_manager

# Import API Routers
from backend.api.system import router as system_router
from backend.api.auth import router as auth_router
from backend.api.studies import router as studies_router
from backend.api.analysis import router as analysis_router
from backend.api.viewer import router as viewer_router
from backend.api.verdict import router as verdict_router
from backend.api.models import router as models_router
from backend.api.experiments import router as experiments_router
from backend.api.datasets import router as datasets_router
from backend.api.reports import router as reports_router
from backend.api.admin import router as admin_router
from backend.api.settings import router as settings_router
from backend.api.nodules import router as nodules_router
from backend.api.longitudinal import router as longitudinal_router
from backend.api.radiologist_review import router as reviews_router
from backend.api.audit import router as audit_router
from backend.api.drift import router as drift_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("backend.main")

# Auto-create all tables on startup
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_SUBTITLE,
    version="2.0.0-production",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount REST Routers under /api
app.include_router(system_router, prefix=settings.API_V1_STR)
app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(studies_router, prefix=settings.API_V1_STR)
app.include_router(analysis_router, prefix=settings.API_V1_STR)
app.include_router(viewer_router, prefix=settings.API_V1_STR)
app.include_router(verdict_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(experiments_router, prefix=settings.API_V1_STR)
app.include_router(datasets_router, prefix=settings.API_V1_STR)
app.include_router(reports_router, prefix=settings.API_V1_STR)
app.include_router(admin_router, prefix=settings.API_V1_STR)
app.include_router(settings_router, prefix=settings.API_V1_STR)
app.include_router(nodules_router, prefix=settings.API_V1_STR)
app.include_router(longitudinal_router, prefix=settings.API_V1_STR)
app.include_router(reviews_router, prefix=settings.API_V1_STR)
app.include_router(audit_router, prefix=settings.API_V1_STR)
app.include_router(drift_router, prefix=settings.API_V1_STR)

# Real-Time WebSocket endpoint
@app.websocket("/api/ws/analysis/{analysis_id}")
async def analysis_websocket_endpoint(websocket: WebSocket, analysis_id: str):
    await ws_manager.connect(websocket, analysis_id)
    try:
        while True:
            # Client can ping or send client commands
            data = await websocket.receive_text()
            # Echo heartbeat
            await websocket.send_text(json.dumps({"type": "PONG", "received": data}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket, analysis_id)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
