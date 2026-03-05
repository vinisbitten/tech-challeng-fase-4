ALERT_RULES = {
    "cirurgia": {
        "classes": ["Cautery", "Suction", "Allis"],
        "pose_check": False,
        "severity": "CRITICO",
    },
    "consulta": {
        "classes": ["comdor"],
        "pose_check": True,
        "severity": "ALTO",
    },
    "fisioterapia": {
        "classes": [
            "Arm_Raise_Incorrect",
            "Knee_Extension_Incorrect",
            "Sit_To_Stand_Incorrect",
        ],
        "pose_check": False,
        "severity": "MEDIO",
    },
    "triagem": {
        "classes": [],
        "pose_check": True,
        "severity": "CRITICO",
    },
}


def evaluate_alert(video_result: dict, audio_emotion: str, transcription: str) -> dict:
    context = video_result.get("context", "consulta")
    rules = ALERT_RULES.get(context, ALERT_RULES["consulta"])
    detections = video_result.get("detections", [])
    pose = video_result.get("pose")

    detected_classes = [d["class"].lower() for d in detections]
    visual_alert = any(cls.lower() in detected_classes for cls in rules["classes"])  # ← fix

    audio_alert = str(audio_emotion).lower() in ["alto", "crítico", "critico", "fear", "anger", "sadness", "disgust"]

    alert_keywords = ["dor", "para", "não quero", "medo", "socorro", "ajuda"]
    text_alert = any(kw in transcription.lower() for kw in alert_keywords)

    pose_alert = False
    if rules["pose_check"] and pose:
        pose_alert = _check_protective_pose(pose)

    if context == "triagem" and not pose_alert:
        yolo_detections = [d for d in detections if d.get("confidence", 0) >= 0.7]
        pose_alert = len(yolo_detections) > 0

    triggered = visual_alert or (audio_alert and text_alert) or pose_alert
    confidence = round(sum([visual_alert, audio_alert, text_alert, pose_alert]) / 4, 2)

    return {
        "alert": triggered,
        "severity": rules["severity"] if triggered else "NORMAL",
        "confidence": confidence,
        "context": context,
        "signals": {
            "visual": visual_alert,
            "audio_emotion": audio_alert,
            "text": text_alert,
            "pose": pose_alert,
        },
    }


def _check_protective_pose(landmarks: list) -> bool:
    try:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist = landmarks[15]
        right_wrist = landmarks[16]
        nose = landmarks[0]
        shoulders_raised = abs(left_shoulder[1] - nose[1]) < 0.15
        wrists_crossed = (
            left_wrist[0] > right_wrist[0] and left_shoulder[0] < right_shoulder[0]
        )
        return shoulders_raised or wrists_crossed
    except Exception:
        return False
