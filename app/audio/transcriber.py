import whisper
import os
from datetime import datetime


class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)

    def transcribe(self, audio_path: str) -> dict:
        result = self.model.transcribe(audio_path, language="pt")

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "text": seg["text"].strip(),
            })

        return {
            "audio": os.path.basename(audio_path),
            "transcribed_at": datetime.now().isoformat(),
            "language": result.get("language", "pt"),
            "text": result["text"].strip(),
            "segments": segments,
        }
