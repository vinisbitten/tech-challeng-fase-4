from fastapi import FastAPI, UploadFile, File, Form
from app.audio.transcriber import transcribe_audio
from app.audio.emotion import detect_emotion
from app.video.detector import VideoDetector
from app.video.report import generate_pdf_report
from app.fusion.alert import evaluate_alert
import tempfile, cv2, os
from datetime import datetime
from collections import Counter
from enum import Enum
import os

REPORTS_DIR = "data/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

app = FastAPI(title="Clinical Video Analysis API")

class VideoContext(str, Enum):
    cirurgia = "cirurgia"
    consulta = "consulta"
    fisioterapia = "fisioterapia"
    triagem = "triagem"


@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    context: VideoContext = Form(VideoContext.consulta),  # cirurgia | consulta | fisioterapia | triagem
):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    try:
        transcription = transcribe_audio(tmp_path)
        transcription_text = transcription.get("text", "")
        emotion = detect_emotion(tmp_path, transcription_text)

        detector = VideoDetector(context=context)
        cap = cv2.VideoCapture(tmp_path)
        all_detections, all_poses = [], []
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = round(total_frames / fps, 2)

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result = detector.analyze_frame(frame)
            timestamp = round(frame_idx / fps, 2)
            for det in result["detections"]:
                det["timestamp"] = timestamp
            all_detections.extend(result["detections"])
            if result["pose"]:
                all_poses.append(result["pose"])
            frame_idx += 1

        cap.release()

    finally:
        try:
            os.unlink(tmp_path)
        except PermissionError:
            pass

    # Contagem por classe
    detections_by_class = dict(Counter(d["class"] for d in all_detections))

    video_result = {
        "video": video.filename,
        "analyzed_at": datetime.now().isoformat(),
        "duration_seconds": duration_seconds,
        "context": context,
        "detections": all_detections,
        "detections_by_class": detections_by_class,
        "total_detections": len(all_detections),
        "pose": all_poses[0] if all_poses else None,
    }

    # Alerta fusão áudio+vídeo
    emotion_label = emotion.get("emotion_analysis", {}).get("risk_level", "BAIXO")
    alert = evaluate_alert(video_result, emotion_label, transcription_text)

    # Nível de risco consolidado
    risk_level = alert.get("severity", "NORMAL")
    video_result["risk_level"] = risk_level

    # Alertas no formato esperado pelo report.py
    alerts_list = []
    if alert.get("alert"):
        for det in all_detections[:5]:
            alerts_list.append({
                "timestamp": det.get("timestamp", 0.0),
                "type": det.get("class", "unknown"),
                "severity": risk_level,
                "confidence": det.get("confidence", 0.0),
            })
    video_result["alerts"] = alerts_list

    # Gera PDF
    report_path = os.path.join(
        REPORTS_DIR,
        f"report_{context.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    )
    generate_pdf_report(video_result, report_path)
    
    return {
        "context": context,
        "transcription": transcription,
        "emotion": emotion,
        "detections": all_detections[:10],
        "detections_by_class": detections_by_class,
        "alert": alert,
        "report": report_path,
    }
