from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from app.video.router import router as video_router
import shutil
import os
import uuid

load_dotenv()

app = FastAPI(
    title="FemHealth Multimodal AI",
    description="Sistema de analise multimodal para saude da mulher",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(video_router)

UPLOAD_DIR = "data/samples"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def root():
    return {"status": "ok", "message": "FemHealth Multimodal AI esta rodando!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze/video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Formato de video invalido.")

    file_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        from app.video.detector import VideoAnalyzer
        from app.video.report import generate_pdf_report

        analyzer = VideoAnalyzer()
        result = analyzer.analyze_video(file_path)

        report_path = file_path.replace(".mp4", "_report.pdf")
        generate_pdf_report(result, report_path)

        result["report_pdf"] = report_path
        return result
    finally:
        os.remove(file_path)


@app.post("/analyze/audio")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp3", ".wav", ".m4a", ".ogg")):
        raise HTTPException(status_code=400, detail="Formato de audio invalido.")

    file_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        from app.audio.transcriber import AudioTranscriber
        from app.audio.emotion import EmotionAnalyzer

        transcriber = AudioTranscriber()
        transcription = transcriber.transcribe(file_path)

        emotion_analyzer = EmotionAnalyzer()
        emotion_result = emotion_analyzer.analyze(file_path, transcription["text"])

        return {**transcription, **emotion_result}
    finally:
        os.remove(file_path)


@app.post("/analyze/text")
async def analyze_text(payload: dict):
    text = payload.get("text", "")
    if not text:
        raise HTTPException(status_code=400, detail="Texto nao informado.")

    from app.text.analizer import TextAnalyzer
    analyzer = TextAnalyzer()
    return analyzer.analyze(text)


@app.post("/analyze/multimodal")
async def analyze_multimodal(
    video: UploadFile = File(None),
    audio: UploadFile = File(None),
    text: str = None
):
    results = {}

    if video:
        video_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{video.filename}"
        with open(video_path, "wb") as f:
            shutil.copyfileobj(video.file, f)
        from app.video.detector import VideoAnalyzer
        results["video"] = VideoAnalyzer().analyze_video(video_path)
        os.remove(video_path)

    if audio:
        audio_path = f"{UPLOAD_DIR}/{uuid.uuid4()}_{audio.filename}"
        with open(audio_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        from app.audio.transcriber import AudioTranscriber
        from app.audio.emotion import EmotionAnalyzer
        transcription = AudioTranscriber().transcribe(audio_path)
        emotion = EmotionAnalyzer().analyze(audio_path, transcription["text"])
        results["audio"] = {**transcription, **emotion}
        os.remove(audio_path)

    if text:
        from app.text.analizer import TextAnalyzer
        results["text"] = TextAnalyzer().analyze(text)

    from app.fusion.alert import FusionAnalyzer
    results["fusion"] = FusionAnalyzer().fuse(results)

    return results
