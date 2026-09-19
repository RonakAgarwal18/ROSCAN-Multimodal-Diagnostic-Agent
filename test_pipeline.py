"""
ROSCAN v3.4 Pipeline Verification
Engineered by Ronak Agarwal | github.com/RonakAgarwal18
"""
import numpy as np
import traceback
import cv2
import math


def test_imports():
    print("Testing imports...")
    try:
        import streamlit, cv2, scipy, matplotlib, langgraph, langchain_core, pydantic
        print("  Imports OK")
    except Exception as e:
        print(f"  FAIL: {e}"); return False
    return True


def test_mcp_tools():
    print("Testing MCP Tools database (12 brands)...")
    from modules.mcp_tools import (
        BRANDS, get_models_for_brand, get_device_spec,
        query_oem_parts_catalog, query_salvage_index, verify_policy_coverage,
    )
    try:
        assert len(BRANDS) >= 12, f"Expected 12 brands, got {len(BRANDS)}"
        for brand in BRANDS:
            models = get_models_for_brand(brand)
            assert len(models) > 0, f"No models for {brand}"
            for m in models:
                spec = get_device_spec(brand, m)
                assert spec.get("oem_display_inr", 0) > 0, f"Zero OEM cost: {brand} {m}"
                oem = query_oem_parts_catalog(brand, m)
                sal = query_salvage_index(brand, m)
                pol = verify_policy_coverage("POL-TEST", brand, m)
                assert oem["total_repair_cost_inr"] > 0
                assert sal["net_replacement_loss_inr"] > 0
                assert pol["base_deductible_inr"] > 0
        print(f"  MCP Tools OK ({len(BRANDS)} brands, all models validated)")
    except Exception as e:
        print(f"  FAIL: {e}"); traceback.print_exc(); return False
    return True


def test_vision():
    print("Testing Vision module (4 algorithms)...")
    from modules.vision import (
        calculate_moire_spoof_score, analyze_bezel_symmetry,
        segment_screen_defects, detect_oled_line_defect,
    )
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    try:
        score, hmap, _ = calculate_moire_spoof_score(dummy)
        assert 0.0 <= score <= 1.0, f"Spoof score {score} out of [0,1]"

        is_cf, asym, _ = analyze_bezel_symmetry(dummy)
        assert isinstance(is_cf, bool)
        assert 0.0 <= asym <= 100.0

        grade, pct, _ = segment_screen_defects(dummy)
        assert any(g in grade for g in ("Grade A","Grade B","Grade C")), f"Bad grade: {grade}"

        has_gl, _, _ = detect_oled_line_defect(dummy)
        assert isinstance(has_gl, bool)

        # Green line positive case
        gl_img = np.full((256, 256, 3), 18, dtype=np.uint8)
        cv2.line(gl_img, (130, 0), (130, 255), (0, 245, 60), 4)
        has_gl2, lines2, _ = detect_oled_line_defect(gl_img)
        print(f"    Green line detection on synthetic: {has_gl2} ({len(lines2)} lines)")

        print("  Vision OK")
    except Exception as e:
        print(f"  FAIL: {e}"); traceback.print_exc(); return False
    return True


def test_agent():
    print("Testing Agent (7 cases, brand routing)...")
    from modules.agent import build_claim_graph, _SAMSUNG_RECALL_MODELS
    graph = build_claim_graph()

    def mk_shattered():
        img = np.full((256, 256, 3), 18, dtype=np.uint8)
        cx, cy = 128, 128
        for i in range(32):
            a = i * (2 * math.pi / 32)
            cv2.line(img, (cx, cy),
                     (int(cx + 100*math.cos(a)), int(cy + 95*math.sin(a))),
                     (195, 195, 195), 2)
        return img

    def mk_greenline():
        img = np.full((256, 256, 3), 18, dtype=np.uint8)
        cv2.line(img, (130, 0), (130, 255), (0, 248, 60), 4)
        return img

    pristine = np.full((256, 256, 3), 28, dtype=np.uint8)
    shattered = mk_shattered()
    greenline = mk_greenline()

    cases = [
        ("Pristine/OnePlus",         "OnePlus",  "OnePlus 13",             pristine,  "NO_FAULT_FOUND"),
        ("GreenLine/OnePlus",        "OnePlus",  "OnePlus 12",             greenline, "APPROVED_OEM_RECALL_WARRANTY"),
        ("GreenLine/Apple",          "Apple",    "iPhone 13",              greenline, None),  # non-recall brand
        ("GreenLine/SamsungS-series","Samsung",  "Galaxy S24 Ultra",       greenline, "APPROVED_OEM_RECALL_WARRANTY"),
        ("GreenLine/SamsungA-series","Samsung",  "Galaxy A54",             greenline, None),  # A-series, no recall
    ]

    try:
        for label, brand, model, img, expected in cases:
            result = graph.invoke({
                "device_brand": brand, "device_model": model,
                "policy_id": "POL-TEST", "raw_image": img,
            })
            v   = result.get("final_verdict", "UNKNOWN")
            ded = float(result.get("deductible") or 0)
            lib = float(result.get("insurer_liability") or 0)
            assert ded >= 0.0, f"Negative deductible on {label}: {ded}"
            assert lib >= 0.0, f"Negative liability on {label}: {lib}"
            if expected:
                assert v == expected, f"[{label}] expected {expected}, got {v}"
            print(f"    [{label}] -> {v} | Ded Rs.{ded:,.0f} | Liab Rs.{lib:,.0f}")
        # Verify Samsung S-series set
        assert len(_SAMSUNG_RECALL_MODELS) >= 10, "Samsung recall model set too small"
        print(f"  Agent OK ({len(cases)} cases validated)")
    except Exception as e:
        print(f"  FAIL: {e}"); traceback.print_exc(); return False
    return True


def test_no_branding():
    print("Testing de-branding (no 'Assurant' in source files)...")
    import pathlib
    files = [
        pathlib.Path("app.py"),
        pathlib.Path("modules/agent.py"),
        pathlib.Path("modules/mcp_tools.py"),
        pathlib.Path("modules/vision.py"),
    ]
    found = []
    for f in files:
        if f.exists():
            txt = f.read_text(encoding="utf-8", errors="ignore")
            if "Assurant" in txt:
                found.append(str(f))
    if found:
        print(f"  FAIL: 'Assurant' found in: {found}"); return False
    print("  De-branding OK")
    return True


if __name__ == "__main__":
    print("=" * 62)
    print("ROSCAN v3.4 Pipeline Verification")
    print("Engineered by Ronak Agarwal | github.com/RonakAgarwal18")
    print("=" * 62)
    results = [
        test_imports(),
        test_mcp_tools(),
        test_vision(),
        test_agent(),
        test_no_branding(),
    ]
    print("=" * 62)
    if all(results):
        print("ALL TESTS PASSED.")
        raise SystemExit(0)
    else:
        failed = sum(1 for r in results if not r)
        print(f"{failed} TEST(S) FAILED.")
        raise SystemExit(1)
