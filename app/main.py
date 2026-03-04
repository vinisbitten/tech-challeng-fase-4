from fastapi import FastAPI, UploadFile, File, Form
from app.audio.transcriber import transcribe_audio
from app.audio.emotion import detect_emotion
from app.video.detector import VideoDetector
from app.video.report import generate_report
from app.fusion.alert import evaluate_alert
import tempfile, cv2, os

app = FastAPI(title="Clinical Video Analysis API")

@app.post("/analyze")
async def analyze(
    video: UploadFile = File(...),
    context: str = Form("consulta")  # cirurgia | consulta | fisioterapia | triagem
):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(await video.read())
        tmp_path = tmp.name

    # Extrai áudio e analisa
    transcription = transcribe_audio(tmp_path)
    emotion = detect_emotion(tmp_path)

    # Analisa frames do vídeo
    detector = VideoDetector(context=context)
    cap = cv2.VideoCapture(tmp_path)
    all_detections, all_poses = [], []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = detector.analyze_frame(frame)
        all_detections.extend(result["detections"])
        if result["pose"]:
            all_poses.append(result["pose"])
    cap.release()
    os.unlink(tmp_path)

    video_result = {
        "context": context,
        "detections": all_detections,
        "pose": all_poses[0] if all_poses else None
    }

    alert = evaluate_alert(video_result, emotion, transcription)
    report_path = generate_report(video_result, alert, transcription, emotion)

    return {
        "context": context,
        "transcription": transcription,
        "emotion": emotion,
        "detections": all_detections[:10],
        "alert": alert,
        "report": report_path
    }
