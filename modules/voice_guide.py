from typing import Dict

def get_visual_guidance_prompt(lighting_confidence: float, angle_confidence: float) -> Dict[str, str]:
    """
    Returns dynamic vernacular visual guidance prompts based on confidence metrics.
    """
    if lighting_confidence < 0.5:
        return {
            "english": "Please move to a well-lit area to avoid shadows.",
            "hinglish": "Kripya acchi lighting wali jagah par jayein taaki shadows na aayein."
        }
    elif angle_confidence < 0.6:
        return {
            "english": "Please tilt the phone to 30-degrees to detect screen reflections.",
            "hinglish": "Phone ko 30-degree tilt kijiye reflection detect karne ke liye."
        }
    else:
        return {
            "english": "Perfect. Please clean the display and inspect it carefully.",
            "hinglish": "Display clean karke inspect karein."
        }
