# src/claude_service.py
"""
Claude AI integration service for PulmoTrust.

Provides async helpers to generate clinical reports, explain XAI findings,
summarise robustness sweeps, and hold free-form clinical chat sessions —
all powered by the Anthropic Claude API.

Usage:
    from src.claude_service import ClaudeService
    svc = ClaudeService()
    report = await svc.generate_clinical_report(diagnosis_data)
"""

import asyncio
import logging
from typing import Any, Dict, List, Optional

import anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# System prompt shared across all requests
# ──────────────────────────────────────────────
_SYSTEM_PROMPT = """You are a highly experienced clinical AI assistant specialising in
thoracic oncology and pulmonary radiology. You work alongside the PulmoTrust AI system,
an explainable deep-learning platform for lung cancer detection on CT scans.

Your role:
- Provide clear, accurate, evidence-based clinical interpretations.
- Always flag uncertainty when model confidence is low (< 0.75) or when
  robustness degradation is high (> 0.15).
- Use precise medical terminology but also include plain-language explanations.
- Never diagnose definitively — always recommend specialist review.
- Adhere to clinical best practices and patient safety principles.
- Format responses in clean Markdown with appropriate headings and bullet points.
"""


class ClaudeService:
    """Async wrapper around the Anthropic client for PulmoTrust AI."""

    def __init__(self) -> None:
        if not ANTHROPIC_API_KEY:
            logger.warning(
                "ANTHROPIC_API_KEY is not set. Claude features will be unavailable. "
                "Add your key to the .env file."
            )
            self._client: Optional[anthropic.AsyncAnthropic] = None
        else:
            self._client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    def _check_client(self) -> None:
        if self._client is None:
            raise ValueError(
                "Anthropic API key not configured. "
                "Set ANTHROPIC_API_KEY in your .env file."
            )

    # ──────────────────────────────────────────────
    # 1. Clinical Report Generation
    # ──────────────────────────────────────────────
    async def generate_clinical_report(
        self,
        diagnosis: Dict[str, Any],
        patient_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate a full clinical narrative report from classification results.

        Args:
            diagnosis: Dict with keys: predicted_class, confidence, probabilities,
                       perturbation, robustness_degradation, explanation_consistency,
                       architecture, inference_time_ms
            patient_context: Optional free-text clinical context from the clinician.

        Returns:
            Dict with 'report' (Markdown str), 'model', 'input_tokens', 'output_tokens'
        """
        self._check_client()

        pred = diagnosis.get("predicted_class", "Unknown")
        conf = diagnosis.get("confidence", 0.0)
        probs = diagnosis.get("probabilities", {})
        rd = diagnosis.get("robustness_degradation", 0.0)
        ec = diagnosis.get("explanation_consistency", 1.0)
        arch = diagnosis.get("architecture", "hybrid")
        inf_ms = diagnosis.get("inference_time_ms", 0.0)
        perturbation = diagnosis.get("perturbation", "none")

        # Format probability distribution
        prob_lines = "\n".join(
            f"  - **{cls}**: {prob:.1%}" for cls, prob in probs.items()
        )

        risk_flag = ""
        if conf < 0.75:
            risk_flag = "\n> ⚠️ **Low confidence prediction** — recommend manual review."
        if rd and rd > 0.15:
            risk_flag += "\n> ⚠️ **High robustness degradation** — prediction may be unreliable."

        context_section = (
            f"\n\n**Clinician Context:** {patient_context}"
            if patient_context
            else ""
        )

        user_message = f"""## CT Lung Cancer Classification Result — Please Generate Clinical Report

**Primary Finding:** {pred}
**Model Confidence:** {conf:.1%}
**Architecture:** {arch.upper()}
**Inference Time:** {inf_ms:.1f} ms
**Perturbation Applied:** {perturbation}
**Robustness Degradation (RD):** {rd:.3f}
**Explanation Consistency (EC):** {ec:.3f}

**Probability Distribution:**
{prob_lines}
{risk_flag}
{context_section}

---
Please generate a structured clinical report with the following sections:
1. **Executive Summary** — one-paragraph overview
2. **Radiological Findings** — CT interpretation based on the classification
3. **Differential Diagnosis** — ranked differential with reasoning
4. **Risk Assessment** — malignancy risk level (Low / Intermediate / High)
5. **AI Model Reliability** — interpret confidence, RD, and EC scores
6. **Clinical Recommendation** — next steps for the clinician
7. **Limitations & Caveats** — model uncertainty, what this AI cannot do
"""

        response = await self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        report_text = response.content[0].text
        return {
            "report": report_text,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    # ──────────────────────────────────────────────
    # 2. XAI Explanation
    # ──────────────────────────────────────────────
    async def explain_xai_findings(
        self,
        xai_type: str,
        predicted_class: str,
        confidence: float,
        probabilities: Dict[str, float],
        architecture: str = "hybrid",
    ) -> Dict[str, Any]:
        """
        Generate a plain-language clinical explanation of XAI visualisation results.

        Args:
            xai_type: One of 'gradcam', 'gradcam_pp', 'attention', 'lime', 'shap'
            predicted_class: Model's prediction
            confidence: Model confidence score
            probabilities: Full probability distribution
            architecture: Model architecture used

        Returns:
            Dict with 'explanation' (Markdown str), 'model', 'input_tokens', 'output_tokens'
        """
        self._check_client()

        xai_descriptions = {
            "gradcam": (
                "Gradient-weighted Class Activation Mapping (Grad-CAM). "
                "Highlights regions in the CT scan whose pixel intensities most strongly "
                "influenced the model's prediction. Red/hot regions = high importance."
            ),
            "gradcam_pp": (
                "Grad-CAM++ (improved Grad-CAM). Provides sharper, more localised "
                "activation maps compared to standard Grad-CAM, better capturing "
                "fine-grained diagnostic features."
            ),
            "attention": (
                "Vision Transformer Attention Rollout. Shows which image patches "
                "the ViT model attended to across all transformer layers. "
                "Bright areas = high cumulative attention weight."
            ),
            "lime": (
                "Local Interpretable Model-agnostic Explanations (LIME). "
                "Identifies superpixel regions that positively (green) or negatively "
                "(red) contributed to the prediction locally."
            ),
            "shap": (
                "SHapley Additive exPlanations (SHAP). Assigns each pixel a fair "
                "contribution score based on game theory, showing feature importance "
                "across the entire prediction."
            ),
        }

        xai_desc = xai_descriptions.get(
            xai_type.lower(),
            f"Explainability method: {xai_type}"
        )

        prob_lines = "\n".join(
            f"  - {cls}: {prob:.1%}" for cls, prob in probabilities.items()
        )

        user_message = f"""## XAI Visualisation Explanation Request

**XAI Method:** {xai_type.upper()}
**Method Description:** {xai_desc}

**Model Prediction:** {predicted_class} ({confidence:.1%} confidence)
**Architecture:** {architecture.upper()}
**Full Probability Distribution:**
{prob_lines}

---
Please explain:
1. **What the Heatmap Shows** — anatomically, what regions are highlighted and why they matter
2. **Clinical Significance** — what these highlighted regions suggest in the context of {predicted_class}
3. **XAI Method Limitations** — known issues or artefacts with {xai_type.upper()} specifically
4. **Trustworthiness Assessment** — does the model appear to be "looking" at clinically relevant regions?
5. **Actionable Insight** — what should the radiologist pay close attention to?

Keep language clear for both radiologists and oncologists.
"""

        response = await self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return {
            "explanation": response.content[0].text,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    # ──────────────────────────────────────────────
    # 3. Robustness Sweep Summary
    # ──────────────────────────────────────────────
    async def explain_robustness(
        self,
        sweep_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Summarise robustness sweep results and flag clinical reliability concerns.

        Args:
            sweep_data: Dict matching RobustnessSweepResponse schema

        Returns:
            Dict with 'summary' (Markdown str), 'model', 'input_tokens', 'output_tokens'
        """
        self._check_client()

        original_class = sweep_data.get("original_class", "Unknown")
        original_conf = sweep_data.get("original_confidence", 0.0)
        mean_rd = sweep_data.get("mean_robustness_degradation", 0.0)
        mean_ec = sweep_data.get("mean_explanation_consistency", 0.0)
        perturbations = sweep_data.get("perturbations", [])

        pert_lines = []
        for p in perturbations:
            pert_lines.append(
                f"  - **{p.get('display_name', p.get('name', '?'))}**: "
                f"Predicted={p.get('predicted_class', '?')} "
                f"({p.get('confidence', 0):.1%}), "
                f"RD={p.get('robustness_degradation', 0):.3f}, "
                f"EC={p.get('explanation_consistency', 0):.3f}"
            )
        pert_block = "\n".join(pert_lines) or "  No perturbation data available."

        user_message = f"""## Robustness Sweep Analysis — Clinical Reliability Assessment

**Original Prediction:** {original_class} ({original_conf:.1%} confidence)
**Mean Robustness Degradation (RD):** {mean_rd:.3f}
**Mean Explanation Consistency (EC):** {mean_ec:.3f}

**Perturbation Results:**
{pert_block}

---
Please generate a robustness analysis report covering:
1. **Overall Stability Assessment** — is the model clinically reliable under common imaging artefacts?
2. **Most Dangerous Perturbations** — which noise types caused the biggest confidence drop or class flip?
3. **Class Flip Analysis** — did any perturbation change the diagnosis? Clinical implications?
4. **Explanation Stability** — are XAI heatmaps consistent across perturbations?
5. **Clinical Deployment Recommendation** — is this model safe to use in clinical settings based on this sweep?
6. **Suggested Mitigations** — what preprocessing or data augmentation could improve robustness?
"""

        response = await self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )

        return {
            "summary": response.content[0].text,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

    # ──────────────────────────────────────────────
    # 4. Free-form Clinical Chat
    # ──────────────────────────────────────────────
    async def chat_about_case(
        self,
        messages: List[Dict[str, str]],
        case_context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Multi-turn clinical Q&A about the current case.

        Args:
            messages: List of {"role": "user"|"assistant", "content": str}
            case_context: Optional dict with current diagnosis data for context injection

        Returns:
            Dict with 'reply' (str), 'model', 'input_tokens', 'output_tokens'
        """
        self._check_client()

        # Build a context preamble if case data is provided
        context_prefix = ""
        if case_context:
            pred = case_context.get("predicted_class", "Unknown")
            conf = case_context.get("confidence", 0.0)
            rd = case_context.get("robustness_degradation", 0.0)
            arch = case_context.get("architecture", "hybrid")
            context_prefix = (
                f"[Current Case Context — CT scan classified as **{pred}** "
                f"with {conf:.1%} confidence using the {arch.upper()} model. "
                f"Robustness Degradation: {rd:.3f}]\n\n"
            )

        # Inject context into the first user message if not already there
        api_messages = []
        for i, msg in enumerate(messages):
            content = msg["content"]
            if i == 0 and context_prefix and context_prefix not in content:
                content = context_prefix + content
            api_messages.append({"role": msg["role"], "content": content})

        response = await self._client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=CLAUDE_MAX_TOKENS,
            system=_SYSTEM_PROMPT,
            messages=api_messages,
        )

        return {
            "reply": response.content[0].text,
            "model": response.model,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }


# ──────────────────────────────────────────────
# Module-level singleton (lazy, thread-safe for FastAPI)
# ──────────────────────────────────────────────
_claude_service: Optional[ClaudeService] = None


def get_claude_service() -> ClaudeService:
    """FastAPI dependency / singleton accessor."""
    global _claude_service
    if _claude_service is None:
        _claude_service = ClaudeService()
    return _claude_service
