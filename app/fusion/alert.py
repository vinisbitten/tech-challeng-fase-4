from typing import Optional

ALERT_RULES = {
    "cirurgia": {
        "classes": ["bleeding", "hemorrhage", "anomaly"],
        "pose_check": False,
        "severity": "CRITICO"
    },
    "consulta": {
        "classes": ["discomfort", "fear", "pain"],
        "pose_check": True,
        "severity": "ALTO"
    },
    "fisioterapia": {
        "classes": ["wrong_posture", "fall_risk"],
        "pose_check": True,
        "severity": "MEDIO"
    },
    "triagem": {
        "classes": ["protective_gesture", "flinch", "covering"],
        "pose_check": True,
        "severity": "CRITICO"
    }
}

def evaluate_alert(video_result: dict, audio_emotion: str, transcription: str) -> dict:
    context = video_result.get("context", "consulta")
    rules = ALERT_RULES.get(context, ALERT_RULES["consulta"])
    detections = video_result.get("detections", [])
    pose = video_result.get("pose")

    detected_classes = [d["class"].lower() for d in detections]
    visual_alert = any(cls in detected_classes for cls in rules["classes"])

    # Sinais de alerta no áudio
    audio_alert = audio_emotion in ["fear", "anger", "sadness", "disgust"]

    # Sinais no texto
    alert_keywords = ["dor", "para", "não quero", "medo", "socorro", "ajuda"]
    text_alert = any(kw in transcription.lower() for kw in alert_keywords)

    # Pose suspeita (braços cruzados/encolhido) para triagem e consulta
    pose_alert = False
    if rules["pose_check"] and pose:
        pose_alert = _check_protective_pose(pose)

    triggered = visual_alert or (audio_alert and text_alert) or pose_alert
    confidence = sum([visual_alert, audio_alert, text_alert, pose_alert]) / 4

    return {
        "alert": triggered,
        "severity": rules["severity"] if triggered else "NORMAL",
        "confidence": round(confidence, 2),
        "context": context,
        "signals": {
            "visual": visual_alert,
            "audio_emotion": audio_alert,
            "text": text_alert,
            "pose": pose_alert
        }
    }

def _check_protective_pose(landmarks: list) -> bool:
    """Verifica postura protetiva: ombros elevados ou braços cruzados."""
    try:
        # landmarks: [x, y, z, vis] — índices MediaPipe
        left_shoulder  = landmarks[11]
        right_shoulder = landmarks[12]
        left_wrist     = landmarks[15]
        right_wrist    = landmarks[16]
        nose           = landmarks[0]

        # Ombros encolhidos: y do ombro próximo ao nariz
        shoulders_raised = abs(left_shoulder[1] - nose[1]) < 0.15

        # Braços cruzados: pulsos invertidos em relação aos ombros
        wrists_crossed = (left_wrist[0] > right_wrist[0] and
                          left_shoulder[0] < right_shoulder[0])

        return shoulders_raised or wrists_crossed
    except Exception:
        return False
