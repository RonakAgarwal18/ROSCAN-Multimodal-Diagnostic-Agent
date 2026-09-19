import numpy as np
import traceback

def test_imports():
    print("Testing imports...")
    try:
        import streamlit
        import cv2
        import scipy
        import matplotlib
        import langgraph
        import langchain_core
        import pydantic
        print("Imports OK")
    except Exception as e:
        print(f"Import Failed: {e}")
        return False
    return True

def test_vision():
    print("Testing Vision Module...")
    from modules.vision import calculate_moire_spoof_score, analyze_bezel_symmetry, segment_screen_defects
    
    dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    try:
        score, heatmap = calculate_moire_spoof_score(dummy_img)
        assert 0.0 <= score <= 1.0, "Spoof score out of bounds"
        
        is_cf, asym = analyze_bezel_symmetry(dummy_img)
        
        grade, defect_pct, blended = segment_screen_defects(dummy_img)
        assert grade in ["Grade A", "Grade B", "Grade C"], "Invalid cosmetic grade"
        
        print("Vision OK")
    except Exception as e:
        print(f"Vision Failed: {e}")
        traceback.print_exc()
        return False
    return True

def test_acoustics():
    print("Testing Acoustics Module...")
    from modules.acoustics import run_acoustic_sweep, compute_stft_diagnostics
    try:
        buffer = run_acoustic_sweep(44100, 0.5)
        status, fig = compute_stft_diagnostics(buffer, 44100)
        assert isinstance(status, str), "Status must be string"
        print("Acoustics OK")
    except Exception as e:
        print(f"Acoustics Failed: {e}")
        traceback.print_exc()
        return False
    return True

def test_agent():
    print("Testing LangGraph Agent...")
    from modules.agent import build_claim_graph
    
    try:
        graph = build_claim_graph()
        dummy_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        
        from modules.acoustics import run_acoustic_sweep
        buffer = run_acoustic_sweep(44100, 0.5)
        
        initial_state = {
            "device_model": "IPHONE-13-PRO",
            "policy_id": "POL-991238",
            "raw_image": dummy_img,
            "audio_buffer": buffer,
            "audio_sample_rate": 44100
        }
        
        final_state = graph.invoke(initial_state)
        
        assert "final_verdict" in final_state, "Agent did not produce a verdict"
        print(f"Agent OK. Verdict: {final_state['final_verdict']}")
    except Exception as e:
        print(f"Agent Failed: {e}")
        traceback.print_exc()
        return False
    return True

if __name__ == "__main__":
    print("Starting ROSCAN Pipeline Verification...")
    v1 = test_imports()
    v2 = test_vision()
    v3 = test_acoustics()
    v4 = test_agent()
    
    if v1 and v2 and v3 and v4:
        print("\\nALL TESTS PASSED.")
        exit(0)
    else:
        print("\\nSOME TESTS FAILED.")
        exit(1)
