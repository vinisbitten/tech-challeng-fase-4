from fastapi import APIRouter, UploadFile, File, HTTPException
import uuid, shutil, os
from app.video.analizer import VideoAnalyzer
from app.video.schemas import VideoAnalysisResponse

router = APIRouter(prefix="/video", tags=["Video Analysis"])

UPLOAD_DIR = "data/samples"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/analyze", response_model=VideoAnalysisResponse)
async def analyze_video(
    file: UploadFile = File(...),
    save_output: bool = False,
):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Formato invalido. Use mp4, avi ou mov.")

    filename = f"{uuid.uuid4()}_{file.filename}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    output_path = os.path.join(UPLOAD_DIR, f"annotated_{filename}") if save_output else None

    analyzer = VideoAnalyzer()
    result = analyzer.analyze_video(filepath, output_path=output_path)
    return result
