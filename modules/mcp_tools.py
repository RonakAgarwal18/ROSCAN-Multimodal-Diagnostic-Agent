from pydantic import BaseModel, Field

class OEMPartsQuery(BaseModel):
    model_id: str = Field(description="Device model identifier (e.g., 'IPHONE-13-PRO')")
    defect_type: str = Field(description="Type of defect identified (e.g., 'SCREEN_CRACK')")

class SalvageIndexQuery(BaseModel):
    model_id: str = Field(description="Device model identifier")
    cosmetic_grade: str = Field(description="Assessed cosmetic grade ('Grade A', 'Grade B', 'Grade C')")

class PolicyCoverageQuery(BaseModel):
    policy_id: str = Field(description="User's insurance policy identifier")
    peril: str = Field(description="Type of incident (e.g., 'ACCIDENTAL_DAMAGE')")

def query_oem_parts_catalog(model_id: str, defect_type: str) -> dict:
    """
    Returns certified OEM display/chassis costs, labor rates, and dispatch turnaround.
    """
    # Mocking MCP tool behavior
    base_costs = {
        "SCREEN_CRACK": 279.0,
        "LIQUID_DAMAGE": 549.0,
        "BURN_IN": 199.0
    }
    cost = base_costs.get(defect_type, 150.0)
    return {
        "tool": "query_oem_parts_catalog",
        "model_id": model_id,
        "defect_type": defect_type,
        "certified_parts_cost": cost,
        "labor_rate_per_hour": 85.0,
        "estimated_labor_hours": 1.5,
        "total_repair_cost": cost + (85.0 * 1.5),
        "dispatch_turnaround_days": 3
    }

def query_salvage_index(model_id: str, cosmetic_grade: str) -> dict:
    """
    Queries circular secondary market scrap value, refurbished resell valuation, and E-waste recycling credit.
    """
    # Mocking MCP tool behavior
    base_market_value = 800.0  # Assumed pristine value
    
    grade_multiplier = {
        "Grade A": 0.85,
        "Grade B": 0.60,
        "Grade C": 0.30
    }
    salvage_multiplier = grade_multiplier.get(cosmetic_grade, 0.1)
    
    salvage_value = base_market_value * salvage_multiplier
    
    return {
        "tool": "query_salvage_index",
        "model_id": model_id,
        "cosmetic_grade": cosmetic_grade,
        "refurbished_resell_valuation": salvage_value * 1.2,
        "secondary_market_scrap_value": salvage_value,
        "ewaste_recycling_credit": 25.0
    }

def verify_policy_coverage(policy_id: str, peril: str) -> dict:
    """
    Returns deductible, aggregate limits, and exclusion validity.
    """
    # Mocking MCP tool behavior
    return {
        "tool": "verify_policy_coverage",
        "policy_id": policy_id,
        "peril": peril,
        "is_covered": True,
        "deductible_payable": 99.0,
        "aggregate_limit": 1500.0,
        "exclusions_triggered": False
    }
