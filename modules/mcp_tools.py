"""
ROSCAN MCP Tools — Indian Market Device Pricing & Policy Registry
All figures in INR. Sources: Cashify Q3-2026, OEM India service rate cards.
Engineered by Ronak Agarwal | github.com/RonakAgarwal18
"""
from __future__ import annotations
from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# Tier definitions (used for models without individually catalogued OEM pricing)
# ─────────────────────────────────────────────────────────────────────────────
TIER_ULTRA    = "ultra_premium"      # >75k
TIER_PREMIUM  = "premium"            # 35k-75k
TIER_MID      = "mid_range"          # 18k-35k
TIER_BUDGET   = "budget"             # <18k

TIER_PRICING: dict[str, dict[str, Any]] = {
    TIER_ULTRA: {
        "oem_display_inr": 33000.0, "labor_inr": 2500.0,
        "cashify_resale_inr": 75000.0, "scrap_inr": 22000.0,
        "pricing_note": "Estimated from Ultra-Premium segment tier (OEM quote pending)",
    },
    TIER_PREMIUM: {
        "oem_display_inr": 15000.0, "labor_inr": 1500.0,
        "cashify_resale_inr": 35000.0, "scrap_inr": 10000.0,
        "pricing_note": "Estimated from Premium segment tier (OEM quote pending)",
    },
    TIER_MID: {
        "oem_display_inr": 7500.0, "labor_inr": 1000.0,
        "cashify_resale_inr": 17000.0, "scrap_inr": 5000.0,
        "pricing_note": "Estimated from Mid-Range segment tier (OEM quote pending)",
    },
    TIER_BUDGET: {
        "oem_display_inr": 4500.0, "labor_inr": 700.0,
        "cashify_resale_inr": 10000.0, "scrap_inr": 3000.0,
        "pricing_note": "Estimated from Budget segment tier (OEM quote pending)",
    },
}


def _t(tier: str, deduct: float, recall: bool = False,
       recall_ded: float = 0.0, recall_pol: str = "") -> dict[str, Any]:
    """Build a model spec from a named tier + policy overrides."""
    base = dict(TIER_PRICING[tier])
    base["base_deductible_inr"] = deduct
    base["oem_recall_eligible"] = recall
    base["recall_deductible_inr"] = recall_ded
    base["recall_policy"] = recall_pol
    return base


def _m(display: float, labor: float, resale: float, scrap: float,
       deduct: float, recall: bool = False, recall_ded: float = 0.0,
       recall_pol: str = "", note: str = "") -> dict[str, Any]:
    """Build a model spec from exact catalogued figures."""
    return {
        "oem_display_inr": display, "labor_inr": labor,
        "cashify_resale_inr": resale, "scrap_inr": scrap,
        "base_deductible_inr": deduct,
        "oem_recall_eligible": recall,
        "recall_deductible_inr": recall_ded,
        "recall_policy": recall_pol,
        "pricing_note": note or "OEM certified service rate card",
    }


_OP_RECALL  = "OnePlus India Lifetime Free Screen Replacement Advisory"
_SAM_RECALL = "Samsung India Special Screen Replacement Program (S-Series)"

# ─────────────────────────────────────────────────────────────────────────────
# Master device registry
# ─────────────────────────────────────────────────────────────────────────────
DEVICE_DB: dict[str, dict[str, Any]] = {

    # ── Apple ─────────────────────────────────────────────────────────────────
    "Apple": {
        "iPhone 18 Pro Max":   _m(38000,2500,100000,35000,4000),
        "iPhone 18 Pro":       _m(36000,2500, 90000,31000,3999),
        "iPhone 18":           _m(29000,2000, 68000,23000,3499),
        "iPhone 17 Pro Max":   _m(37900,2500, 95000,32000,3500),
        "iPhone 17 Pro":       _m(35500,2500, 86000,29000,3499),
        "iPhone 17":           _m(28000,2000, 64000,21000,3299),
        "iPhone 16 Pro Max":   _m(37000,2500, 88000,30000,3500),
        "iPhone 16 Pro":       _m(34500,2500, 80000,27000,3499),
        "iPhone 16 Plus":      _m(29000,2000, 60000,20000,3299),
        "iPhone 16":           _m(27900,2000, 55000,18500,2999),
        "iPhone 15 Pro Max":   _m(37900,2500, 95000,32000,3500),
        "iPhone 15 Pro":       _m(35500,2500, 82000,28000,3499),
        "iPhone 15":           _m(27900,2000, 52000,18000,2999),
        "iPhone 14 Pro":       _m(28500,2000, 48000,16000,2999),
        "iPhone 14":           _m(23000,1800, 38000,12500,2499),
        "iPhone 13":           _m(21500,1800, 34000,11000,2499),
        "iPhone 12":           _m(18500,1600, 24000, 8000,1999),
        "iPhone SE (3rd Gen)": _m(12000,1200, 16000, 5000,1499),
    },

    # ── Samsung ──────────────────────────────────────────────────────────────
    "Samsung": {
        "Galaxy S26 Ultra":   _m(35000,2000, 95000,33000,3500,True,649,_SAM_RECALL),
        "Galaxy S26+":        _m(29000,1800, 75000,25000,2999,True,649,_SAM_RECALL),
        "Galaxy S26":         _m(24000,1600, 62000,20000,2999,True,649,_SAM_RECALL),
        "Galaxy S25 Ultra":   _m(33000,2000, 90000,30000,3299,True,649,_SAM_RECALL),
        "Galaxy S25+":        _m(27000,1800, 70000,23000,2999,True,649,_SAM_RECALL),
        "Galaxy S25":         _m(22500,1600, 55000,18000,2499,True,649,_SAM_RECALL),
        "Galaxy S24 Ultra":   _m(23500,1500, 72000,24000,2999,True,649,_SAM_RECALL),
        "Galaxy S24+":        _m(20000,1500, 58000,19000,2499,True,649,_SAM_RECALL),
        "Galaxy S24":         _m(18500,1400, 48000,15000,2249,True,649,_SAM_RECALL),
        "Galaxy S23 Ultra":   _m(22000,1500, 60000,20000,2999,True,649,_SAM_RECALL),
        "Galaxy S23":         _m(17000,1400, 38000,12000,2249,True,649,_SAM_RECALL),
        "Galaxy S22 Ultra":   _m(20000,1500, 48000,16000,2499,True,649,_SAM_RECALL),
        "Galaxy S22":         _m(14200,1200, 28000, 9000,1799,True,649,_SAM_RECALL),
        "Galaxy S21 FE":      _m(13000,1200, 22000, 7000,1499,True,649,_SAM_RECALL),
        "Galaxy Z Fold 7":    _m(38000,2800, 95000,32000,4500),
        "Galaxy Z Fold 6":    _m(36000,2800, 85000,28000,4299),
        "Galaxy Z Flip 7":    _m(22000,2000, 48000,16000,2999),
        "Galaxy Z Flip 6":    _m(20000,2000, 40000,13000,2499),
        "Galaxy A55":         _m( 7800, 900, 20000, 6500, 999),
        "Galaxy A54":         _m( 6900, 800, 18000, 5500, 999),
        "Galaxy A35":         _m( 6200, 800, 15000, 4500, 799),
        "Galaxy M35":         _m( 5800, 800, 14000, 4000, 799),
        "Galaxy F55":         _m( 6000, 800, 14500, 4200, 799),
    },

    # ── OnePlus ───────────────────────────────────────────────────────────────
    "OnePlus": {
        "OnePlus 14":      _m(18000,1400, 46000,15000,1799,True,0,_OP_RECALL),
        "OnePlus 13":      _m(16500,1300, 42000,13500,1699,True,0,_OP_RECALL),
        "OnePlus 13R":     _m(10000,1000, 26000, 8000,1199,True,0,_OP_RECALL),
        "OnePlus 12":      _m(15800,1200, 38000,12000,1499,True,0,_OP_RECALL),
        "OnePlus 12R":     _m( 9500,1000, 24000, 7500,1199,True,0,_OP_RECALL),
        "OnePlus 11":      _m(14500,1200, 32000,10000,1499,True,0,_OP_RECALL),
        "OnePlus 11R":     _m( 8500,1000, 20000, 6000,1099,True,0,_OP_RECALL),
        "OnePlus 10 Pro":  _m(13500,1000, 24000, 7500,1299,True,0,_OP_RECALL),
        "OnePlus 10R":     _m( 8200,1000, 18000, 5500,1099,True,0,_OP_RECALL),
        "OnePlus 9 Pro":   _m(12500, 950, 20000, 6500,1199,True,0,_OP_RECALL),
        "OnePlus 9":       _m(11800, 900, 16000, 5000, 999,True,0,_OP_RECALL),
        "OnePlus 8T":      _m(10500, 900, 13000, 4000, 999,True,0,_OP_RECALL),
        "OnePlus Nord 4":  _m( 7500, 850, 18000, 5500, 899,True,0,_OP_RECALL),
        "OnePlus Nord CE 4": _m(5800, 750, 14000, 4000, 799,True,0,_OP_RECALL),
    },

    # ── Xiaomi / Redmi ────────────────────────────────────────────────────────
    "Xiaomi / Redmi": {
        "Xiaomi 15":              _m(19500,1400, 46000,14500,1899),
        "Xiaomi 14 Ultra":        _m(28000,1800, 65000,21000,2999),
        "Xiaomi 14":              _m(16500,1200, 40000,12000,1499),
        "Xiaomi 13 Pro":          _m(15000,1200, 32000,10000,1499),
        "Redmi Note 15 Pro+":     _m( 6500, 700, 18000, 5500, 899),
        "Redmi Note 14 Pro+":     _m( 6000, 700, 16000, 4800, 849),
        "Redmi Note 13 Pro+":     _m( 5200, 600, 15000, 4000, 799),
        "Redmi Note 13":          _m( 4200, 600, 11000, 3200, 699),
        "Redmi 13C":              _m( 3200, 500,  8000, 2500, 499),
    },

    # ── POCO ──────────────────────────────────────────────────────────────────
    "POCO": {
        "POCO F6 Pro":   _m(10500, 950, 24000, 7500,1099),
        "POCO F6":       _m( 8500, 900, 19000, 5800, 999),
        "POCO X6 Pro":   _m( 7200, 800, 17000, 5000, 899),
        "POCO X6":       _m( 6000, 750, 14000, 4000, 799),
        "POCO M6 Pro":   _m( 4500, 600, 10500, 3200, 599),
    },

    # ── Realme ────────────────────────────────────────────────────────────────
    "Realme": {
        "Realme GT 7 Pro":    _m(14000,1200, 36000,11000,1499),
        "Realme GT 6":        _m(10500,1000, 26000, 8000,1099),
        "Realme 13 Pro+":     _m( 7000, 800, 18000, 5500, 899),
        "Realme 12 Pro+":     _m( 5600, 600, 16000, 4500, 799),
        "Realme P1 Pro":      _m( 5200, 600, 14000, 4000, 749),
        "Realme Narzo 70 Pro":_m( 4500, 600, 11000, 3200, 649),
    },

    # ── Vivo ──────────────────────────────────────────────────────────────────
    "Vivo": {
        "Vivo X100 Ultra":  _m(25000,1800, 58000,18000,2499),
        "Vivo X100 Pro":    _m(18200,1400, 44000,14000,1999),
        "Vivo X100":        _m(14500,1200, 34000,10500,1699),
        "Vivo V40 Pro":     _m( 9500,1000, 24000, 7500,1099),
        "Vivo V30 Pro":     _m( 8500,1000, 20000, 6000, 999),
        "Vivo T3 Pro":      _m( 6500, 800, 15000, 4500, 849),
        "Vivo Y200":        _m( 4800, 650, 11000, 3200, 649),
    },

    # ── iQOO ──────────────────────────────────────────────────────────────────
    "iQOO": {
        "iQOO 13":           _m(14500,1200, 38000,12000,1499),
        "iQOO 12":           _m(12500,1100, 30000, 9500,1299),
        "iQOO Neo 9 Pro":    _m( 7800, 800, 23000, 6500, 999),
        "iQOO Z9 Turbo":     _m( 6200, 750, 16000, 4800, 849),
        "iQOO Z9":           _m( 5200, 700, 13000, 3800, 749),
    },

    # ── OPPO ──────────────────────────────────────────────────────────────────
    "OPPO": {
        "OPPO Find X8 Pro":  _m(22000,1600, 52000,17000,2499),
        "OPPO Reno 12 Pro":  _m( 9500,1000, 24000, 7500,1099),
        "OPPO Reno 11":      _m( 8000, 900, 20000, 6000, 999),
        "OPPO F27 Pro+":     _m( 6500, 800, 16000, 4800, 849),
        "OPPO A3 Pro":       _m( 4800, 650, 11000, 3200, 649),
    },

    # ── Motorola ─────────────────────────────────────────────────────────────
    "Motorola": {
        "Edge 50 Ultra":    _m(13000,1100, 30000, 9000,1299),
        "Edge 50 Pro":      _m( 9400, 900, 21000, 6000, 999),
        "Edge 50 Fusion":   _m( 7200, 850, 16000, 4800, 849),
        "Edge 40":          _m( 8500, 900, 18000, 5500, 999),
        "Moto G85":         _m( 5200, 700, 12000, 3500, 699),
        "Razr 50 Ultra":    _m(20000,1800, 44000,14000,2299),
    },

    # ── Google Pixel ──────────────────────────────────────────────────────────
    "Google Pixel": {
        "Pixel 10 Pro":  _m(28000,2000, 65000,21000,2999),
        "Pixel 9 Pro":   _m(25000,2000, 55000,18000,2699),
        "Pixel 9":       _m(18000,1600, 40000,12500,1999),
        "Pixel 8a":      _m(12000,1200, 26000, 8000,1499),
        "Pixel 8 Pro":   _m(22000,1800, 46000,15000,2499),
        "Pixel 7a":      _m(10500,1100, 22000, 6500,1299),
    },

    # ── Nothing / CMF ────────────────────────────────────────────────────────
    "Nothing / CMF": {
        "Nothing Phone (3)":     _m(12500,1100, 30000, 9000,1299),
        "Nothing Phone (2a) Plus":_m(7500, 850, 17000, 5000, 899),
        "Nothing Phone (2)":     _m( 9500,1000, 20000, 6200,1099),
        "CMF Phone 1":           _m( 4500, 650, 10000, 3000, 599),
    },
}

BRANDS: list[str] = list(DEVICE_DB.keys())


def get_models_for_brand(brand: str) -> list[str]:
    return list(DEVICE_DB.get(brand, {}).keys())


def get_device_spec(brand: str, model: str) -> dict[str, Any]:
    return DEVICE_DB.get(brand, {}).get(model, {})


# ─────────────────────────────────────────────────────────────────────────────
# MCP Tool Wrappers
# ─────────────────────────────────────────────────────────────────────────────

def query_oem_parts_catalog(brand: str, model: str) -> dict[str, Any]:
    spec = get_device_spec(brand, model)
    display = spec.get("oem_display_inr", 7500.0)
    labor   = spec.get("labor_inr", 1000.0)
    return {
        "tool": "query_oem_parts_catalog",
        "brand": brand, "model": model,
        "certified_display_assembly_inr": display,
        "authorized_labor_inr": labor,
        "total_repair_cost_inr": display + labor,
        "dispatch_turnaround_days": 3,
        "service_partner": f"Authorized {brand} India Service Center",
        "oem_recall_eligible": spec.get("oem_recall_eligible", False),
        "recall_policy": spec.get("recall_policy", "N/A"),
        "recall_deductible_inr": spec.get("recall_deductible_inr", 0.0),
        "pricing_note": spec.get("pricing_note", "OEM certified service rate card"),
    }


def query_salvage_index(brand: str, model: str) -> dict[str, Any]:
    spec   = get_device_spec(brand, model)
    resale = spec.get("cashify_resale_inr", 17000.0)
    scrap  = spec.get("scrap_inr", 5000.0)
    return {
        "tool": "query_salvage_index",
        "brand": brand, "model": model,
        "cashify_fair_market_resale_inr": resale,
        "salvage_scrap_recovery_inr": scrap,
        "net_replacement_loss_inr": resale - scrap,
        "ewaste_recycling_credit_inr": 350.0,
        "data_source": "Cashify Circular Economy Index Q3-2026",
    }


def verify_policy_coverage(policy_id: str, brand: str, model: str) -> dict[str, Any]:
    spec    = get_device_spec(brand, model)
    deduct  = spec.get("base_deductible_inr", 999.0)
    return {
        "tool": "verify_policy_coverage",
        "policy_id": policy_id, "brand": brand, "model": model,
        "base_deductible_inr": deduct,
        "aggregate_limit_inr": 150000.0,
        "subrogation_eligible": spec.get("oem_recall_eligible", False),
        "policy_section_ref": "Section 3.1 / 4.2 — ROSCAN India CEP",
    }
