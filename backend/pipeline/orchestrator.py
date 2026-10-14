"""
backend/pipeline/orchestrator.py
===================================
Master ML Pipeline Orchestrator for the Lung Cancer AI Platform.

Coordinates all pipeline stages in a dependency-aware DAG:

  Stage 0 → Data validation & DICOM QA
  Stage 1 → CT Preprocessing  (HU windowing, isotropic resampling)
  Stage 2 → Segmentation       (UNet3D lung + nodule masks)
  Stage 3 → Detection          (RetinaNet3D candidate generation)
  Stage 4 → Classification     (ResNet3D / DenseNet3D / SwinTransformer3D)
  Stage 5 → Radiomics          (PyRadiomics IBSI feature extraction)
  Stage 6 → Ensemble           (DL + Radiomics + Clinical fusion)
  Stage 7 → XAI                (3D Grad-CAM, IG, Attention Rollout)
  Stage 8 → Uncertainty        (MC Dropout predictive entropy)
  Stage 9 → Calibration        (Temperature scaling)
  Stage 10→ Report generation  (structured clinical JSON report)

Usage
-----
>>> from backend.pipeline import PipelineOrchestrator
>>> orch = PipelineOrchestrator.from_config()
>>> report = await orch.run(dicom_dir="/data/cases/CASE_001")
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.monitoring import get_logger, MetricsRegistry

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stage enumeration
# ---------------------------------------------------------------------------
class PipelineStage(str, Enum):
    DICOM_QA        = "dicom_qa"
    PREPROCESSING   = "preprocessing"
    SEGMENTATION    = "segmentation"
    DETECTION       = "detection"
    CLASSIFICATION  = "classification"
    RADIOMICS       = "radiomics"
    ENSEMBLE        = "ensemble"
    XAI             = "xai"
    UNCERTAINTY     = "uncertainty"
    CALIBRATION     = "calibration"
    REPORT          = "report"


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------
@dataclass
class PipelineContext:
    """Mutable state passed through every stage."""
    case_id:        str
    dicom_dir:      Path
    stage_outputs:  Dict[str, Any] = field(default_factory=dict)
    errors:         List[str]      = field(default_factory=list)
    warnings:       List[str]      = field(default_factory=list)
    start_time:     float          = field(default_factory=time.monotonic)
    enabled_stages: List[PipelineStage] = field(
        default_factory=lambda: list(PipelineStage)
    )

    def set(self, stage: PipelineStage, data: Any) -> None:
        self.stage_outputs[stage.value] = data

    def get(self, stage: PipelineStage, default: Any = None) -> Any:
        return self.stage_outputs.get(stage.value, default)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self.start_time


@dataclass
class PipelineReport:
    """Final structured clinical report returned by the orchestrator."""
    case_id:         str
    status:          str          # "success" | "partial" | "failed"
    elapsed_s:       float
    qa_passed:       bool
    predictions:     Dict[str, Any] = field(default_factory=dict)
    ensemble_result: Dict[str, Any] = field(default_factory=dict)
    uncertainty:     Dict[str, Any] = field(default_factory=dict)
    xai_artifacts:   Dict[str, Any] = field(default_factory=dict)
    radiomics:       Dict[str, Any] = field(default_factory=dict)
    errors:          List[str]      = field(default_factory=list)
    warnings:        List[str]      = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id":         self.case_id,
            "status":          self.status,
            "elapsed_s":       round(self.elapsed_s, 3),
            "qa_passed":       self.qa_passed,
            "predictions":     self.predictions,
            "ensemble":        self.ensemble_result,
            "uncertainty":     self.uncertainty,
            "xai_artifacts":   self.xai_artifacts,
            "radiomics":       self.radiomics,
            "errors":          self.errors,
            "warnings":        self.warnings,
        }


# ---------------------------------------------------------------------------
# Pipeline Orchestrator
# ---------------------------------------------------------------------------
class PipelineOrchestrator:
    """
    Async, stage-by-stage ML pipeline controller.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory containing model weight files (.pt / .pth).
    device : str
        Torch device string — "cuda", "cpu", or "cuda:0".
    enabled_stages : list[PipelineStage] | None
        Subset of stages to run. Defaults to all stages.
    skip_on_error : bool
        If True, a stage failure adds an error to context but the
        pipeline continues. If False, it raises immediately.
    """

    def __init__(
        self,
        checkpoint_dir: Path = Path("checkpoints"),
        device: str = "cpu",
        enabled_stages: Optional[List[PipelineStage]] = None,
        skip_on_error: bool = True,
    ) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.device = device
        self.enabled_stages = enabled_stages or list(PipelineStage)
        self.skip_on_error = skip_on_error

        # Stage runners (lazy import — avoids PyTorch load at import time)
        self._stage_runners: Dict[PipelineStage, Any] = {}
        log.info(
            "pipeline_orchestrator_init",
            device=device,
            checkpoint_dir=str(checkpoint_dir),
            stages=[s.value for s in self.enabled_stages],
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, config: Optional[Dict[str, Any]] = None) -> "PipelineOrchestrator":
        """Create from a config dict or environment variables."""
        import os
        cfg = config or {}
        return cls(
            checkpoint_dir=Path(cfg.get("checkpoint_dir", os.getenv("CHECKPOINT_DIR", "checkpoints"))),
            device=cfg.get("device", os.getenv("TORCH_DEVICE", "cpu")),
            skip_on_error=cfg.get("skip_on_error", True),
        )

    # ------------------------------------------------------------------
    # Lazy stage loader
    # ------------------------------------------------------------------
    def _get_runner(self, stage: PipelineStage):
        if stage in self._stage_runners:
            return self._stage_runners[stage]
        runner = self._build_runner(stage)
        self._stage_runners[stage] = runner
        return runner

    def _build_runner(self, stage: PipelineStage):
        """Lazy-import and instantiate the stage runner."""
        if stage == PipelineStage.DICOM_QA:
            from backend.dicom.validator import DicomValidator
            return DicomValidator()

        if stage == PipelineStage.PREPROCESSING:
            from backend.preprocessing.ct_pipeline import CTPreprocessingPipeline
            return CTPreprocessingPipeline(device=self.device)

        if stage == PipelineStage.SEGMENTATION:
            from backend.segmentation.inference import SegmentationInference
            return SegmentationInference(
                checkpoint_dir=self.checkpoint_dir, device=self.device
            )

        if stage == PipelineStage.DETECTION:
            from backend.pipeline.stages.detection import DetectionStage
            return DetectionStage(
                checkpoint_dir=self.checkpoint_dir, device=self.device
            )

        if stage == PipelineStage.CLASSIFICATION:
            from backend.pipeline.stages.classification import ClassificationStage
            return ClassificationStage(
                checkpoint_dir=self.checkpoint_dir, device=self.device
            )

        if stage == PipelineStage.RADIOMICS:
            from backend.models.radiomics.extractor import RadiomicsExtractor
            return RadiomicsExtractor()

        if stage == PipelineStage.ENSEMBLE:
            from backend.pipeline.stages.ensemble import EnsembleStage
            return EnsembleStage(checkpoint_dir=self.checkpoint_dir)

        if stage == PipelineStage.XAI:
            from backend.pipeline.stages.xai import XAIStage
            return XAIStage(device=self.device)

        if stage == PipelineStage.UNCERTAINTY:
            from backend.uncertainty.mc_dropout import MCDropoutEstimator
            return MCDropoutEstimator(device=self.device)

        if stage == PipelineStage.CALIBRATION:
            from backend.calibration.temperature_scaling import TemperatureScaler
            return TemperatureScaler()

        if stage == PipelineStage.REPORT:
            from backend.pipeline.stages.report import ReportStage
            return ReportStage()

        raise ValueError(f"Unknown stage: {stage}")

    # ------------------------------------------------------------------
    # Core runner
    # ------------------------------------------------------------------
    async def run(
        self,
        dicom_dir: str | Path,
        case_id: Optional[str] = None,
        extra_clinical: Optional[Dict[str, Any]] = None,
    ) -> PipelineReport:
        """
        Execute the full ML pipeline for a single CT case.

        Parameters
        ----------
        dicom_dir : path to folder with DICOM slices (or a .nii.gz file)
        case_id   : optional identifier; auto-generated from path if absent
        extra_clinical : additional clinical metadata (age, sex, smoking_pack_years)

        Returns
        -------
        PipelineReport with all stage outputs
        """
        dicom_path = Path(dicom_dir)
        case_id = case_id or dicom_path.name

        ctx = PipelineContext(
            case_id=case_id,
            dicom_dir=dicom_path,
            enabled_stages=self.enabled_stages,
        )
        if extra_clinical:
            ctx.set("clinical_meta", extra_clinical)

        log.info("pipeline_start", case_id=case_id, dicom_dir=str(dicom_path))
        MetricsRegistry.update_gpu_memory()

        for stage in self.enabled_stages:
            await self._run_stage(stage, ctx)
            if ctx.errors and not self.skip_on_error:
                break

        return self._build_report(ctx)

    async def _run_stage(
        self, stage: PipelineStage, ctx: PipelineContext
    ) -> None:
        t0 = time.monotonic()
        log.info("stage_start", stage=stage.value, case_id=ctx.case_id)
        try:
            runner = self._get_runner(stage)
            # Support both sync and async runners
            if asyncio.iscoroutinefunction(getattr(runner, "run", None)):
                result = await runner.run(ctx)
            else:
                result = await asyncio.to_thread(runner.run, ctx)
            ctx.set(stage, result)
            elapsed = time.monotonic() - t0
            log.info(
                "stage_done",
                stage=stage.value,
                case_id=ctx.case_id,
                elapsed_ms=round(elapsed * 1000, 1),
            )
        except Exception as exc:
            elapsed = time.monotonic() - t0
            msg = f"[{stage.value}] {type(exc).__name__}: {exc}"
            ctx.errors.append(msg)
            log.error(
                "stage_error",
                stage=stage.value,
                case_id=ctx.case_id,
                error=msg,
                elapsed_ms=round(elapsed * 1000, 1),
            )

    def _build_report(self, ctx: PipelineContext) -> PipelineReport:
        qa_out   = ctx.get(PipelineStage.DICOM_QA) or {}
        ens_out  = ctx.get(PipelineStage.ENSEMBLE) or {}
        unc_out  = ctx.get(PipelineStage.UNCERTAINTY) or {}
        xai_out  = ctx.get(PipelineStage.XAI) or {}
        rad_out  = ctx.get(PipelineStage.RADIOMICS) or {}
        cls_out  = ctx.get(PipelineStage.CLASSIFICATION) or {}

        status = (
            "failed" if len(ctx.errors) == len(ctx.enabled_stages)
            else "partial" if ctx.errors
            else "success"
        )

        MetricsRegistry.record_inference(
            model="pipeline",
            outcome=status,
            latency_ms=ctx.elapsed_s * 1000,
        )

        report = PipelineReport(
            case_id=ctx.case_id,
            status=status,
            elapsed_s=ctx.elapsed_s,
            qa_passed=qa_out.get("passed", False),
            predictions=cls_out if isinstance(cls_out, dict) else {},
            ensemble_result=ens_out if isinstance(ens_out, dict) else {},
            uncertainty=unc_out if isinstance(unc_out, dict) else {},
            xai_artifacts=xai_out if isinstance(xai_out, dict) else {},
            radiomics=rad_out if isinstance(rad_out, dict) else {},
            errors=ctx.errors,
            warnings=ctx.warnings,
        )
        log.info(
            "pipeline_complete",
            case_id=ctx.case_id,
            status=status,
            elapsed_s=round(ctx.elapsed_s, 3),
            error_count=len(ctx.errors),
        )
        return report
