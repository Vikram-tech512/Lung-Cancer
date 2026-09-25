"""
backend/websocket/manager.py
============================
Real-time WebSocket event broadcaster for live CT analysis pipelines.
Supports multi-client synchronization and Redis Pub/Sub integration.
"""

from __future__ import annotations
import json
import logging
import asyncio
from typing import Dict, List, Any
from fastapi import WebSocket

logger = logging.getLogger("backend.websocket")

class ConnectionManager:
    def __init__(self):
        # Maps analysis_id -> list of connected WebSockets
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, analysis_id: str):
        await websocket.accept()
        if analysis_id not in self.active_connections:
            self.active_connections[analysis_id] = []
        self.active_connections[analysis_id].append(websocket)
        logger.info(f"Client connected to WebSocket channel for analysis: {analysis_id}")

    def disconnect(self, websocket: WebSocket, analysis_id: str):
        if analysis_id in self.active_connections:
            if websocket in self.active_connections[analysis_id]:
                self.active_connections[analysis_id].remove(websocket)
            if not self.active_connections[analysis_id]:
                del self.active_connections[analysis_id]
        logger.info(f"Client disconnected from analysis channel: {analysis_id}")

    async def broadcast_event(self, analysis_id: str, stage: str, progress: int, payload: Dict[str, Any] = None):
        """Broadcasts real server events to all connected clients."""
        message = {
            "analysis_id": analysis_id,
            "stage": stage,
            "progress_percent": progress,
            "data": payload or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        if analysis_id in self.active_connections:
            dead_sockets = []
            for connection in self.active_connections[analysis_id]:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    logger.warning(f"Error sending message to socket: {e}")
                    dead_sockets.append(connection)
            for dead in dead_sockets:
                self.disconnect(dead, analysis_id)

ws_manager = ConnectionManager()
