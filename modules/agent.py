import numpy as np
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END

# Import mock tools
from .mcp_tools import query_oem_parts_catalog, query_salvage_index, verify_policy_coverage
from .vision import calculate_moire_spoof_score, analyze_bezel_symmetry, segment_screen_defects
from .acoustics import run_acoustic_sweep, compute_stft_diagnostics

class ClaimState(TypedDict):
    device_model: str
    policy_id: str
    raw_image: Optional[np.ndarray]
    audio_buffer: Optional[np.ndarray]
    audio_sample_rate: int
    
    # Processed states
    spoof_score: Optional[float]
    is_counterfeit: Optional[bool]
    defect_grade: Optional[str]
    defect_percentage: Optional[float]
    acoustic_status: Optional[str]
    
    # Economics
    oem_quote: Optional[dict]
    salvage_quote: Optional[dict]
    policy_coverage: Optional[dict]
    
    # Verdict
    final_verdict: Optional[str]
    decision_reasoning: Optional[str]

def verify_authenticity(state: ClaimState) -> ClaimState:
    if state["raw_image"] is not None:
        spoof_score, _ = calculate_moire_spoof_score(state["raw_image"])
        is_counterfeit, _ = analyze_bezel_symmetry(state["raw_image"])
        state["spoof_score"] = spoof_score
        state["is_counterfeit"] = is_counterfeit
    return state

def segment_defects(state: ClaimState) -> ClaimState:
    if state["raw_image"] is not None:
        grade, percentage, _ = segment_screen_defects(state["raw_image"])
        state["defect_grade"] = grade
        state["defect_percentage"] = percentage
    return state

def acoustic_check(state: ClaimState) -> ClaimState:
    if state["audio_buffer"] is not None:
        status, _ = compute_stft_diagnostics(state["audio_buffer"], state["audio_sample_rate"])
        state["acoustic_status"] = status
    return state

def fetch_mcp_economics(state: ClaimState) -> ClaimState:
    state["oem_quote"] = query_oem_parts_catalog(state["device_model"], "SCREEN_CRACK")
    
    grade = state.get("defect_grade", "Grade B")
    state["salvage_quote"] = query_salvage_index(state["device_model"], grade)
    
    state["policy_coverage"] = verify_policy_coverage(state["policy_id"], "ACCIDENTAL_DAMAGE")
    return state

def synthesize_adjudication(state: ClaimState) -> ClaimState:
    # Rule 1: Fraud/Spoofing check
    if state.get("spoof_score", 0.0) > 0.65 or state.get("is_counterfeit", False):
        state["final_verdict"] = "REJECT_FRAUD"
        state["decision_reasoning"] = "Claim rejected due to high likelihood of spoofing or counterfeit replacement panel."
        return state
        
    oem = state.get("oem_quote", {})
    salvage = state.get("salvage_quote", {})
    
    total_repair = oem.get("total_repair_cost", 500.0)
    market_value = salvage.get("refurbished_resell_valuation", 800.0)
    scrap_value = salvage.get("secondary_market_scrap_value", 200.0)
    
    net_replacement = market_value - scrap_value
    
    # Rule 2: Economic comparison
    if total_repair < net_replacement:
        state["final_verdict"] = "APPROVE_REPAIR"
        state["decision_reasoning"] = f"Repair cost (${total_repair:.2f}) is less than net replacement cost (${net_replacement:.2f}). Authorizing OEM repair."
    else:
        state["final_verdict"] = "APPROVE_REPLACEMENT"
        state["decision_reasoning"] = f"Repair cost (${total_repair:.2f}) exceeds net replacement cost (${net_replacement:.2f}). Authorizing full device replacement."
        
    return state

def build_claim_graph() -> StateGraph:
    workflow = StateGraph(ClaimState)
    
    workflow.add_node("verify_authenticity", verify_authenticity)
    workflow.add_node("segment_defects", segment_defects)
    workflow.add_node("acoustic_check", acoustic_check)
    workflow.add_node("fetch_mcp_economics", fetch_mcp_economics)
    workflow.add_node("synthesize_adjudication", synthesize_adjudication)
    
    workflow.add_edge(START, "verify_authenticity")
    workflow.add_edge("verify_authenticity", "segment_defects")
    workflow.add_edge("segment_defects", "acoustic_check")
    workflow.add_edge("acoustic_check", "fetch_mcp_economics")
    workflow.add_edge("fetch_mcp_economics", "synthesize_adjudication")
    workflow.add_edge("synthesize_adjudication", END)
    
    return workflow.compile()
