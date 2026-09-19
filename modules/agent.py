"""
ROSCAN Adjudication Agent — LangGraph State Machine
7-case decision matrix with brand-specific dynamic economics.
Engineered by Ronak Agarwal | github.com/RonakAgarwal18
"""
from __future__ import annotations

import numpy as np
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

from .vision import (
    calculate_moire_spoof_score,
    analyze_bezel_symmetry,
    segment_screen_defects,
    detect_oled_line_defect,
)
from .mcp_tools import query_oem_parts_catalog, query_salvage_index, verify_policy_coverage

# Samsung S-series models that qualify for the free replacement programme
_SAMSUNG_RECALL_MODELS: frozenset[str] = frozenset({
    "Galaxy S26 Ultra","Galaxy S26+","Galaxy S26",
    "Galaxy S25 Ultra","Galaxy S25+","Galaxy S25",
    "Galaxy S24 Ultra","Galaxy S24+","Galaxy S24",
    "Galaxy S23 Ultra","Galaxy S23",
    "Galaxy S22 Ultra","Galaxy S22",
    "Galaxy S21 FE",
})


class ClaimState(TypedDict):
    # Inputs
    device_brand: str
    device_model: str
    policy_id: str
    raw_image: Optional[np.ndarray]

    # Vision outputs
    spoof_score: Optional[float]
    spoof_reason: Optional[str]
    is_counterfeit: Optional[bool]
    bezel_asymmetry_pct: Optional[float]
    bezel_explainer: Optional[str]
    defect_grade: Optional[str]
    defect_percentage: Optional[float]
    has_green_line: Optional[bool]
    defect_overlay: Optional[np.ndarray]

    # Economics
    oem_quote: Optional[dict]
    salvage_quote: Optional[dict]
    policy_coverage: Optional[dict]

    # Adjudication outputs
    final_verdict: Optional[str]
    decision_reasoning: Optional[str]
    deductible: Optional[float]
    insurer_liability: Optional[float]
    repair_cost: Optional[float]


# ─────────────────────────────────────────────────────────────────────────────
# Graph nodes
# ─────────────────────────────────────────────────────────────────────────────

def verify_authenticity(state: ClaimState) -> ClaimState:
    if state.get("raw_image") is not None:
        spoof, _, reason   = calculate_moire_spoof_score(state["raw_image"])
        is_cf, asym, exp   = analyze_bezel_symmetry(state["raw_image"])
        state["spoof_score"]        = spoof
        state["spoof_reason"]       = reason
        state["is_counterfeit"]     = is_cf
        state["bezel_asymmetry_pct"] = asym
        state["bezel_explainer"]    = exp
    return state


def segment_defects(state: ClaimState) -> ClaimState:
    if state.get("raw_image") is not None:
        grade, pct, overlay = segment_screen_defects(state["raw_image"])
        state["defect_grade"]      = grade
        state["defect_percentage"] = pct
        state["defect_overlay"]    = overlay
        has_gl, _, _               = detect_oled_line_defect(state["raw_image"])
        state["has_green_line"]    = has_gl
    return state


def fetch_mcp_economics(state: ClaimState) -> ClaimState:
    brand  = state.get("device_brand", "Samsung")
    model  = state.get("device_model", "Galaxy S22")
    policy = state.get("policy_id",    "POL-IND-0001")
    state["oem_quote"]       = query_oem_parts_catalog(brand, model)
    state["salvage_quote"]   = query_salvage_index(brand, model)
    state["policy_coverage"] = verify_policy_coverage(policy, brand, model)
    return state


def synthesize_adjudication(state: ClaimState) -> ClaimState:
    spoof       = float(state.get("spoof_score")         or 0.0)
    counterfeit = bool(state.get("is_counterfeit")       or False)
    grade       = state.get("defect_grade")              or "Grade A"
    has_gl      = bool(state.get("has_green_line")       or False)
    asym        = float(state.get("bezel_asymmetry_pct") or 0.0)
    brand       = state.get("device_brand",  "Samsung")
    model       = state.get("device_model",  "Galaxy S22")

    oem    = state.get("oem_quote")      or {}
    sal    = state.get("salvage_quote")  or {}
    pol    = state.get("policy_coverage") or {}

    total_repair   = float(oem.get("total_repair_cost_inr",    14200.0))
    net_repl_loss  = float(sal.get("net_replacement_loss_inr", 22500.0))
    base_deduct    = float(pol.get("base_deductible_inr",       1499.0))
    recall_elig    = bool(oem.get("oem_recall_eligible",        False))
    recall_ded     = float(oem.get("recall_deductible_inr",     0.0))
    recall_policy  = str(oem.get("recall_policy",               "OEM Warranty"))

    def _set(verdict: str, rationale: str,
             deduct: float, liability: float, repair: float) -> None:
        state["final_verdict"]      = verdict
        state["decision_reasoning"] = rationale
        state["deductible"]         = max(deduct, 0.0)
        state["insurer_liability"]  = max(liability, 0.0)
        state["repair_cost"]        = max(repair, 0.0)

    # ── Case 6: Screen Replay Fraud ───────────────────────────────────────────
    # Conservative: only trigger on PAPR > 4.5 sigma in ultra-high freq band
    # with geometric symmetry confirmation (spoof_score > 0.65 encodes both).
    if spoof > 0.65:
        _set("REJECT_FRAUD",
             "Periodic pixel grid confirmed via 2D-FFT PAPR analysis (peak >4.5 sigma) "
             "with geometric harmonic symmetry verified in the ultra-high frequency band "
             "(>0.85 Nyquist). Pattern consistent with secondary digital monitor replay attack. "
             "Claim frozen. Case escalated to Special Investigations Unit.",
             0.0, 0.0, 0.0)
        return state

    # ── Case 7: Non-OEM counterfeit panel ─────────────────────────────────────
    # Only trigger on flat, unskewed captures with asymmetry > 20 %
    if counterfeit and asym > 20.0:
        _set("REJECT_UNAUTHORIZED_MOD",
             f"Non-OEM aftermarket display assembly detected on {brand} {model}. "
             f"Chin-to-bezel asymmetry of {asym:.1f}% exceeds the 20% OEM factory tolerance "
             "threshold on a flat, perspective-invariant capture. "
             "Violates Section 4.2: unauthorized component installation voids coverage.",
             0.0, 0.0, 0.0)
        return state

    # ── Case 1: Pristine ──────────────────────────────────────────────────────
    if grade == "Grade A" and not has_gl:
        _set("NO_FAULT_FOUND",
             f"Hardware inspected in pristine factory condition ({brand} {model}). "
             "Zero structural glass fractures or panel anomalies detected. "
             "Claim closed with no loss payable and Rs.0 customer deductible.",
             0.0, 0.0, 0.0)
        return state

    # ── Case 2: Superficial wear ──────────────────────────────────────────────
    if grade == "Grade B" and not has_gl:
        _set("CLAIM_DISMISSED_WEAR_AND_TEAR",
             f"Superficial hairline abrasions detected on {brand} {model}. "
             "Under Section 3.1 of Underwriting Terms, minor cosmetic wear that does not "
             "impair display operation or structural integrity is excluded from coverage.",
             0.0, 0.0, 0.0)
        return state

    # ── Case 4: AMOLED green line on pristine / minor glass — OEM recall ──────
    if has_gl and grade in ("Grade A", "Grade B"):
        # OnePlus Lifetime Screen Warranty (all eligible OnePlus models)
        if brand == "OnePlus" and recall_elig:
            _set("APPROVED_OEM_RECALL_WARRANTY",
                 f"Vertical AMOLED column failure verified on {brand} {model}. "
                 "Adjudicated under the OnePlus India Lifetime Free Screen Replacement Advisory. "
                 "No physical fracture impact points present. Subrogation protocol active: "
                 "100% replacement liability routed directly to OEM authorized service network "
                 "at Rs.0 cost to customer or insurer.",
                 0.0, 0.0, total_repair)
            return state

        # Samsung S-series one-time free replacement
        if brand == "Samsung" and model in _SAMSUNG_RECALL_MODELS and recall_elig:
            _set("APPROVED_OEM_RECALL_WARRANTY",
                 f"AMOLED panel defect verified on {brand} {model}. "
                 "Qualifies under the Samsung India Special Screen Replacement Program. "
                 "Underwriter liability waived under warranty subrogation. "
                 f"Customer nominal administrative fee: Rs.{recall_ded:,.0f}.",
                 recall_ded, 0.0, total_repair)
            return state

        # All other brands — no recall programme, standard repair
        liability = max(total_repair - base_deduct, 0.0)
        _set("APPROVED_REPAIR",
             f"Vertical AMOLED column line confirmed on {brand} {model}. "
             "No active OEM free replacement programme available for this device. "
             f"Claim approved under standard Accidental Damage terms. "
             f"OEM repair authorization: Rs.{total_repair:,.0f}. "
             f"Customer deductible: Rs.{base_deduct:,.0f}.",
             base_deduct, liability, total_repair)
        return state

    # ── Case 5: AMOLED line + cracked glass ───────────────────────────────────
    if has_gl and grade == "Grade C":
        liability = max(total_repair - base_deduct, 0.0)
        _set("APPROVED_REPAIR",
             f"Vertical AMOLED column line detected concurrently with physical fractures on "
             f"{brand} {model}. Physical impact damage voids free OEM recall coverage. "
             f"Claim approved under standard Accidental Damage terms. "
             f"OEM repair authorization: Rs.{total_repair:,.0f}. "
             f"Customer deductible: Rs.{base_deduct:,.0f}.",
             base_deduct, liability, total_repair)
        return state

    # ── Case 3: Accidental glass shatter ─────────────────────────────────────
    if grade == "Grade C":
        if total_repair < net_repl_loss:
            liability = max(total_repair - base_deduct, 0.0)
            _set("APPROVED_REPAIR",
                 f"Structural glass fracture confirmed on {brand} {model}. "
                 f"Authorized OEM repair (Rs.{total_repair:,.0f}) is economically viable "
                 f"vs. net replacement exposure (Rs.{net_repl_loss:,.0f}). "
                 f"Claim authorized under Accidental Damage policy. "
                 f"Customer deductible: Rs.{base_deduct:,.0f} | "
                 f"Net underwriter liability: Rs.{liability:,.0f}.",
                 base_deduct, liability, total_repair)
        else:
            liability = max(net_repl_loss - base_deduct, 0.0)
            _set("APPROVED_REPLACEMENT",
                 f"Structural glass fracture confirmed on {brand} {model}. "
                 f"OEM repair cost (Rs.{total_repair:,.0f}) exceeds net replacement threshold "
                 f"(Rs.{net_repl_loss:,.0f}). Device replacement authorized. "
                 f"Customer deductible: Rs.{base_deduct:,.0f} | "
                 f"Net underwriter liability: Rs.{liability:,.0f}.",
                 base_deduct, liability, net_repl_loss)
        return state

    # Fallback
    _set("NO_FAULT_FOUND",
         "Inspection complete. No actionable claim event detected.", 0.0, 0.0, 0.0)
    return state


# ─────────────────────────────────────────────────────────────────────────────
# Graph builder
# ─────────────────────────────────────────────────────────────────────────────

def build_claim_graph():
    wf = StateGraph(ClaimState)
    wf.add_node("verify_authenticity",    verify_authenticity)
    wf.add_node("segment_defects",        segment_defects)
    wf.add_node("fetch_mcp_economics",    fetch_mcp_economics)
    wf.add_node("synthesize_adjudication", synthesize_adjudication)

    wf.add_edge(START,                    "verify_authenticity")
    wf.add_edge("verify_authenticity",    "segment_defects")
    wf.add_edge("segment_defects",        "fetch_mcp_economics")
    wf.add_edge("fetch_mcp_economics",    "synthesize_adjudication")
    wf.add_edge("synthesize_adjudication", END)

    return wf.compile()
