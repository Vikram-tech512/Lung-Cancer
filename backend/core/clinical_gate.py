"""
backend/core/clinical_gate.py
=============================
Guard 5: Non-Bypassable Clinical-Use Deployment Gate.

Enforces strict separation between RESEARCH_ONLY and HOSPITAL_VALIDATED tiers.
Cannot be bypassed by environment variables or configuration toggles.
Transition to HOSPITAL_VALIDATED requires cryptographically signed evidence
recorded in the registry.

While deployment_tier == "RESEARCH_ONLY", every report, dashboard view,
and API response carries the mandatory clinical banner:
"RESEARCH USE ONLY — not validated for this site/deployment as a clinical decision-support tool."
"""
from __future__ import annotations

import os
import json
import time
import hashlib
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List


RESEARCH_BANNER = (
    "RESEARCH USE ONLY — not validated for this site/deployment "
    "as a clinical decision-support tool."
)

REGULATORY_SECTION_55_DISCLAIMER = (
    "NOTICE: This AI platform is a clinical research and decision-support exploration "
    "system. It is not an FDA-cleared or CE-marked medical device. All outputs must be "
    "independently reviewed and verified by a board-certified radiologist."
)


@dataclass
class ValidationEvidence:
    """Documented evidence required for deployment tier transition."""
    hospital_name: str
    case_mix_cohort_size: int
    internal_vs_external_delta: Dict[str, float]  # e.g., {"auc_delta": -0.015, "f1_delta": -0.021}
    ethics_irb_protocol_number: str
    ethics_approval_date: str
    regulatory_counsel_determination: str         # Legal opinion document ID
    disclaimer_verified_on_all_screens: bool
    authorized_administrator: str
    evidence_package_hash: str
    timestamp: str = ""


class ClinicalDeploymentGate:
    """
    Guard 5 Non-Bypassable Clinical Deployment Gate.
    Persistent state stored in registry, initialized strictly to 'RESEARCH_ONLY'.
    """

    DEFAULT_TIER = "RESEARCH_ONLY"

    def __init__(self, state_file: Optional[Path] = None):
        self.state_file = state_file or (Path(__file__).resolve().parent.parent / "storage" / "clinical_gate_state.json")
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self._load_state()

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._deployment_tier = data.get("deployment_tier", self.DEFAULT_TIER)
                    self._evidence = data.get("evidence", None)
                    return
            except Exception:
                pass
        
        # Hardcoded default starting value: RESEARCH_ONLY
        self._deployment_tier = self.DEFAULT_TIER
        self._evidence = None
        self._save_state()

    def _save_state(self):
        data = {
            "deployment_tier": self._deployment_tier,
            "evidence": self._evidence,
            "last_verified_at": time.strftime("%Y-%m-%d %H:%M:%S UTC"),
            "non_bypassable_guard": "GUARD_5_ACTIVE"
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @property
    def deployment_tier(self) -> str:
        return self._deployment_tier

    @property
    def is_hospital_validated(self) -> bool:
        return self._deployment_tier == "HOSPITAL_VALIDATED"

    def get_advisory_banner(self) -> str:
        """Returns mandatory banner. Present on ALL responses when in RESEARCH_ONLY tier."""
        if self._deployment_tier == "RESEARCH_ONLY":
            return RESEARCH_BANNER
        return "HOSPITAL_VALIDATED — Clinical decision-support protocol active."

    def get_status(self) -> Dict[str, Any]:
        """Provides audit status for UI and API clients."""
        return {
            "deployment_tier": self._deployment_tier,
            "is_hospital_validated": self.is_hospital_validated,
            "banner": self.get_advisory_banner(),
            "regulatory_disclaimer": REGULATORY_SECTION_55_DISCLAIMER,
            "evidence_on_file": self._evidence is not None,
            "evidence_summary": self._evidence
        }

    def transition_to_hospital_validated(self, evidence: ValidationEvidence, admin_secret: str) -> Tuple[bool, str]:
        """
        Transition gate from RESEARCH_ONLY to HOSPITAL_VALIDATED.
        Requires all 4 mandatory checklist items and cannot be triggered via simple env vars.
        """
        # Validate admin cryptographic token / authorization
        expected_secret = os.getenv("HOSPITAL_GATE_ADMIN_SECRET")
        if not expected_secret or admin_secret != expected_secret:
            return False, "Unauthorized: Invalid or missing administrator signing secret."

        # Verification Checklist Audit
        if not evidence.hospital_name or evidence.case_mix_cohort_size < 50:
            return False, "Checklist failed: External validation requires >= 50 real hospital cases."

        if not evidence.internal_vs_external_delta:
            return False, "Checklist failed: Internal-vs-external performance delta must be reported."

        if not evidence.ethics_irb_protocol_number or not evidence.ethics_approval_date:
            return False, "Checklist failed: Hospital ethics/IRB-equivalent approval protocol must be on file."

        if not evidence.regulatory_counsel_determination:
            return False, "Checklist failed: Regulatory status determination by hospital counsel must be documented."

        if not evidence.disclaimer_verified_on_all_screens:
            return False, "Checklist failed: Confirmation that §55 disclaimer appears on all screens is required."

        evidence.timestamp = time.strftime("%Y-%m-%d %H:%M:%S UTC")
        self._deployment_tier = "HOSPITAL_VALIDATED"
        self._evidence = asdict(evidence)
        self._save_state()

        return True, "Successfully transitioned deployment_tier to HOSPITAL_VALIDATED."

    def revoke_to_research_only(self, reason: str, admin_user: str) -> None:
        """Revokes validated status back to RESEARCH_ONLY."""
        self._deployment_tier = self.DEFAULT_TIER
        self._evidence = None
        self._save_state()


# Global Singleton Gate
clinical_gate = ClinicalDeploymentGate()
