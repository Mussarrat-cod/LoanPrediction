"""
xai_engine.py -- Explainable AI engine using SHAP KernelExplainer.

Provides:
  - XAIEngine: wraps a trained model with SHAP-based local/global explanations,
    natural-language adverse-action notices, and actionable recommendations.
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shap

from backend.config import (
    DECISION_THRESHOLD,
    FEATURE_DISPLAY_NAMES,
    PROB_CLIP,
    SHAP_SIGNIFICANCE_CUTOFFS,
    get_risk_tier,
)
from backend.data_layer import engineer_features

logger = logging.getLogger(__name__)


class XAIEngine:
    """Explainable AI engine combining prediction, SHAP explanations,
    natural-language notices, and actionable recommendations.

    Args:
        model: A trained sklearn model (Pipeline or CalibratedClassifierCV)
            that exposes ``predict_proba``.
        background_df: A small DataFrame of representative samples used as
            the SHAP background (KernelExplainer integration point).
        feature_names: Ordered list of feature column names expected by
            the model after feature engineering.
    """

    def __init__(
        self,
        model: Any,
        background_df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
        self.background_df = background_df.copy()
        self.feature_names = feature_names or list(background_df.columns)

        # Build SHAP KernelExplainer
        self._explainer = shap.KernelExplainer(
            self._predict_proba_wrapper, self.background_df
        )
        logger.info(
            "XAIEngine initialised (background=%d rows, features=%d).",
            len(self.background_df), len(self.feature_names),
        )

        # Caches
        self._global_shap: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_proba_wrapper(self, X: np.ndarray) -> np.ndarray:
        """Wrapper that accepts a numpy array or DataFrame and returns
        P(Approved) -- the positive-class probability."""
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.feature_names)
        probs = self.model.predict_proba(X)[:, 1]
        return probs

    def _prepare_input(self, applicant: dict) -> pd.DataFrame:
        """Engineer features for a single applicant dict and align to
        the expected column order."""
        df = engineer_features(applicant)
        # Keep only the columns the model expects, in order
        df = df[self.feature_names]
        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, applicant: dict) -> Dict[str, Any]:
        """Run inference on a single applicant.

        Returns:
            dict with keys: decision, probability, risk_tier, manual_review,
            threshold, engineered_features.
        """
        t0 = time.time()
        df = self._prepare_input(applicant)
        raw_prob = self.model.predict_proba(df)[:, 1][0]
        prob = float(np.clip(raw_prob, PROB_CLIP[0], PROB_CLIP[1]))
        decision = "Approved" if prob >= DECISION_THRESHOLD else "Rejected"
        tier = get_risk_tier(prob)
        manual_review = (
            tier == "Subprime"
            or (decision == "Approved" and tier == "High Risk")
        )

        # Collect engineered feature values for the UI
        eng_df = engineer_features(applicant)
        eng_features = {
            col: float(eng_df[col].iloc[0])
            for col in [
                "loan_to_income", "emi", "debt_income_ratio",
                "income_per_dep", "total_assets", "asset_coverage",
            ]
        }

        latency_ms = round((time.time() - t0) * 1000, 2)
        return {
            "decision": decision,
            "probability": round(prob, 4),
            "risk_tier": tier,
            "manual_review": manual_review,
            "threshold": DECISION_THRESHOLD,
            "latency_ms": latency_ms,
            "engineered_features": eng_features,
        }

    # ------------------------------------------------------------------
    # Local explanation
    # ------------------------------------------------------------------

    def explain_local(self, applicant: dict) -> List[Dict[str, Any]]:
        """Compute per-feature SHAP values for a single applicant.

        Returns a list of dicts sorted by |shap|, each containing:
            feature, display_name, shap_value, feature_value, direction,
            significance.
        """
        df = self._prepare_input(applicant)
        shap_values = self._explainer.shap_values(df.values, nsamples="auto")

        # shap_values may be a list (one per class) or 2-D array
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        sv = np.asarray(sv).flatten()

        max_abs = float(np.max(np.abs(sv))) if np.max(np.abs(sv)) > 0 else 1.0

        contributions: List[Dict[str, Any]] = []
        for i, feat in enumerate(self.feature_names):
            val = float(sv[i])
            raw_val = df.iloc[0, i]
            try:
                feat_val: Any = round(float(raw_val), 4)
            except (ValueError, TypeError):
                feat_val = str(raw_val)  # categorical features
            ratio = abs(val) / max_abs
            sig = _significance_bucket(ratio)
            contributions.append({
                "feature": feat,
                "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                "shap_value": round(val, 6),
                "feature_value": feat_val,
                "direction": "Positive" if val >= 0 else "Negative",
                "significance": sig,
            })

        contributions.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
        return contributions

    # ------------------------------------------------------------------
    # Global explanation
    # ------------------------------------------------------------------

    def explain_global(
        self,
        X_sample: Optional[pd.DataFrame] = None,
        intrinsic_importance: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Compute mean |SHAP| over a sample of rows (cached) plus
        the model's intrinsic importance if provided.

        Args:
            X_sample: Optional DataFrame of test rows to compute SHAP over.
                If None, uses the background set.
            intrinsic_importance: Optional list from model_layer.get_global_importance.

        Returns dict with 'shap_global' and 'intrinsic_importance'.
        """
        if self._global_shap is None:
            data = X_sample if X_sample is not None else self.background_df
            data = data[self.feature_names]
            logger.info("Computing global SHAP over %d samples...", len(data))
            shap_values = self._explainer.shap_values(data.values, nsamples="auto")
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values
            sv = np.asarray(sv)
            mean_abs = np.mean(np.abs(sv), axis=0)

            global_list = []
            for i, feat in enumerate(self.feature_names):
                global_list.append({
                    "feature": feat,
                    "display_name": FEATURE_DISPLAY_NAMES.get(feat, feat),
                    "mean_abs_shap": round(float(mean_abs[i]), 6),
                })
            global_list.sort(key=lambda x: x["mean_abs_shap"], reverse=True)
            self._global_shap = global_list

        return {
            "shap_global": self._global_shap,
            "intrinsic_importance": intrinsic_importance or [],
        }

    # ------------------------------------------------------------------
    # Natural-language explanation
    # ------------------------------------------------------------------

    def generate_nl_explanation(
        self,
        local: List[Dict[str, Any]],
        decision: str,
        tier: str,
        probability: float,
    ) -> str:
        """Generate a regulatory-style adverse-action notice.

        Includes decision, probability, top 3 negative and positive factors.
        """
        neg = [c for c in local if c["direction"] == "Negative"][:3]
        pos = [c for c in local if c["direction"] == "Positive"][:3]

        lines = [
            f"LOAN APPLICATION DECISION: {decision.upper()}",
            f"Approval probability: {probability:.1%} | Risk tier: {tier}",
            "",
        ]

        if neg:
            lines.append("Principal factors AGAINST approval:")
            for i, c in enumerate(neg, 1):
                lines.append(
                    f"  {i}. {c['display_name']} "
                    f"(value: {_fmt_val(c['feature_value'])}, "
                    f"impact: {c['shap_value']:.4f})"
                )
            lines.append("")

        if pos:
            lines.append("Principal factors IN FAVOR of approval:")
            for i, c in enumerate(pos, 1):
                lines.append(
                    f"  {i}. {c['display_name']} "
                    f"(value: {_fmt_val(c['feature_value'])}, "
                    f"impact: {c['shap_value']:.4f})"
                )
            lines.append("")

        if decision == "Rejected":
            lines.append(
                "This decision was made based on the factors above. "
                "You may improve your application by addressing the "
                "negative factors listed."
            )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def generate_recommendations(
        self, local: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate rule-based and SHAP-driven recommendations.

        Only recommends actions on factors the applicant can influence.
        """
        recs: List[Dict[str, str]] = []
        neg = [c for c in local if c["direction"] == "Negative"]

        # Map features to actionable recommendations
        actionable = {
            "cibil_score": {
                "title": "Improve your credit (CIBIL) score",
                "detail": (
                    "Pay existing EMIs on time, reduce credit utilisation, "
                    "and avoid new hard inquiries. A score above 750 "
                    "significantly improves approval odds."
                ),
            },
            "loan_to_income": {
                "title": "Reduce the loan-to-income ratio",
                "detail": (
                    "Consider applying for a smaller loan amount or "
                    "increasing your documented income before applying."
                ),
            },
            "debt_income_ratio": {
                "title": "Lower your debt-to-income ratio",
                "detail": (
                    "Pay off existing debts, choose a longer loan term "
                    "to reduce EMI, or apply for a smaller loan."
                ),
            },
            "emi": {
                "title": "Reduce the estimated monthly instalment",
                "detail": (
                    "Opt for a longer loan term or a smaller principal "
                    "to bring the EMI within affordable limits."
                ),
            },
            "asset_coverage": {
                "title": "Increase asset coverage",
                "detail": (
                    "Add collateral, include a co-signer with assets, "
                    "or build your asset base before applying."
                ),
            },
            "total_assets": {
                "title": "Build your total asset value",
                "detail": (
                    "Accumulate savings, investments, or property to "
                    "demonstrate greater financial stability."
                ),
            },
            "income_per_dep": {
                "title": "Improve income per dependent",
                "detail": (
                    "Increase your income or apply jointly with a "
                    "co-applicant to spread the dependency ratio."
                ),
            },
            "income_annum": {
                "title": "Increase your annual income",
                "detail": (
                    "Document all income sources or consider a "
                    "co-applicant to increase household income."
                ),
            },
            "loan_amount": {
                "title": "Apply for a smaller loan",
                "detail": (
                    "A smaller loan amount improves multiple ratios "
                    "and increases the likelihood of approval."
                ),
            },
            "loan_term": {
                "title": "Consider adjusting the loan term",
                "detail": (
                    "A longer term reduces EMI but increases total "
                    "interest; a shorter term shows repayment capacity."
                ),
            },
        }

        seen = set()
        for c in neg:
            feat = c["feature"]
            if feat in actionable and feat not in seen:
                recs.append({
                    "feature": feat,
                    "title": actionable[feat]["title"],
                    "detail": actionable[feat]["detail"],
                    "impact": c["significance"],
                })
                seen.add(feat)

        return recs

    # ------------------------------------------------------------------
    # SHAP local accuracy check
    # ------------------------------------------------------------------

    def check_shap_accuracy(
        self, applicant: dict, tol: float = 1e-2
    ) -> Dict[str, Any]:
        """Verify: base_value + sum(SHAP) ~ model output."""
        df = self._prepare_input(applicant)
        raw_prob = self._predict_proba_wrapper(df.values)[0]

        shap_values = self._explainer.shap_values(df.values, nsamples="auto")
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values
        sv = np.asarray(sv).flatten()
        base = self._explainer.expected_value
        if isinstance(base, (list, np.ndarray)):
            base = base[1] if len(base) > 1 else base[0]

        reconstructed = float(base) + float(sv.sum())
        diff = abs(reconstructed - float(raw_prob))

        result = {
            "base_value": round(float(base), 6),
            "shap_sum": round(float(sv.sum()), 6),
            "reconstructed": round(reconstructed, 6),
            "model_output": round(float(raw_prob), 6),
            "difference": round(diff, 6),
            "within_tolerance": diff < tol,
        }
        if not result["within_tolerance"]:
            logger.warning(
                "SHAP accuracy check FAILED: diff=%.6f > tol=%.4f", diff, tol
            )
        else:
            logger.info("SHAP accuracy check passed (diff=%.6f).", diff)
        return result


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _significance_bucket(ratio: float) -> str:
    """Map |shap| / max(|shap|) to a significance label."""
    for label, cutoff in SHAP_SIGNIFICANCE_CUTOFFS.items():
        if ratio >= cutoff:
            return label
    return "Low"


def _fmt_val(val: Any) -> str:
    """Format a feature value for display: float with 2 decimals, or string as-is."""
    if isinstance(val, (int, float)):
        return f"{val:.2f}"
    return str(val)

