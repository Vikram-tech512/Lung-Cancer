"""
backend/storage/storage_manager.py
==================================
Object storage manager for high-throughput thoracic CT DICOM series,
NIfTI volumes, 3D segmentation masks, XAI heatmaps, and generated PDF reports.
"""

from __future__ import annotations
import os
import shutil
from pathlib import Path
from typing import Optional, BinaryIO
from backend.core.config import settings

class StorageManager:
    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path or settings.STORAGE_ROOT)
        self.base_path.mkdir(parents=True, exist_ok=True)

    def get_study_dir(self, study_id: str, subfolder: str = "original") -> Path:
        p = self.base_path / "studies" / study_id / subfolder
        p.mkdir(parents=True, exist_ok=True)
        return p

    def save_file(self, study_id: str, subfolder: str, filename: str, content: bytes) -> str:
        folder = self.get_study_dir(study_id, subfolder)
        target = folder / filename
        with open(target, "wb") as f:
            f.write(content)
        return str(target.relative_to(self.base_path).as_posix())

    def get_file_bytes(self, relative_path: str) -> Optional[bytes]:
        full_path = self.base_path / relative_path
        if not full_path.exists():
            return None
        with open(full_path, "rb") as f:
            return f.read()

    def get_absolute_path(self, relative_path: str) -> str:
        return str((self.base_path / relative_path).resolve())

storage_client = StorageManager()
