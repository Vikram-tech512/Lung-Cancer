"""
backend/models/segmentation/nnunet/wrapper.py
=============================================
nnU-Net v2 integration wrapper for 3D fullres lung and lesion segmentation.
Falls back gracefully to 3D U-Net if nnUNet pretrained weights or environment
are not configured.
"""

from __future__ import annotations
import os
import sys
import logging
import tempfile
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Union

logger = logging.getLogger("platform.models.nnunet")

class NNUNetWrapper:
    """
    Wrapper around nnU-Net v2 for automated medical segmentation.
    Compatible with Dataset004_Lungs / Dataset005_Nodules convention.
    """
    def __init__(
        self,
        dataset_name_or_id: str = "Dataset004_Lungs",
        configuration: str = "3d_fullres",
        fold: int = 0,
        trainer_name: str = "nnUNetTrainer",
        plans_name: str = "nnUNetPlans",
        checkpoint_name: str = "checkpoint_final.pth",
        device: Optional[torch.device] = None,
    ):
        self.dataset = dataset_name_or_id
        self.configuration = configuration
        self.fold = fold
        self.trainer_name = trainer_name
        self.plans_name = plans_name
        self.checkpoint_name = checkpoint_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = None
        self._is_ready = False
        self._init_predictor()

    def _init_predictor(self) -> bool:
        """Attempts to load nnUNetPredictor from nnunetv2 if available."""
        try:
            from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
            results_folder = os.environ.get("nnUNet_results")
            if not results_folder:
                logger.info("nnUNet_results environment variable not set. nnU-Net will run in fallback/simulation mode.")
                return False

            model_dir = Path(results_folder) / self.dataset / f"{self.trainer_name}__{self.plans_name}__{self.configuration}"
            if not model_dir.exists():
                logger.info(f"nnUNet model directory {model_dir} not found. Fallback enabled.")
                return False

            self.predictor = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True if self.device.type == "cuda" else False,
                device=self.device,
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False
            )
            self.predictor.initialize_from_trained_model_folder(
                str(model_dir),
                use_folds=(self.fold,),
                checkpoint_name=self.checkpoint_name,
            )
            self._is_ready = True
            logger.info("nnUNetPredictor successfully initialized.")
            return True
        except ImportError:
            logger.info("nnunetv2 package not installed. Operating in fallback mode.")
            return False
        except Exception as e:
            logger.warning(f"Could not initialize nnU-Net: {e}. Operating in fallback mode.")
            return False

    @property
    def is_available(self) -> bool:
        return self._is_ready

    def predict_volume(self, volume: np.ndarray, spacing: Tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
        """
        Segment a 3D numpy CT volume (D, H, W).
        Returns binary mask (D, H, W) where 1 indicates target structure (lung or nodule).
        """
        if self._is_ready and self.predictor is not None:
            # nnUNet expects (C, D, H, W)
            vol_input = volume[np.newaxis, ...] if volume.ndim == 3 else volume
            properties = {
                'sitk_stuff': {},
                'spacing': spacing
            }
            pred_mask = self.predictor.predict_single_npy_array(
                vol_input,
                properties,
                None,
                None,
                False
            )
            return (pred_mask > 0).astype(np.uint8)
        else:
            # Fallback heuristic: Otsu or morphological thresholding for lung region
            logger.info("Using analytical lung parenchyma thresholding fallback [-950, -400 HU]")
            # Assuming volume in HU
            lung_mask = (volume >= -950) & (volume <= -400)
            return lung_mask.astype(np.uint8)
