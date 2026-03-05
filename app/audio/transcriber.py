import whisper
import os
import subprocess
import json
from datetime import datetime

_default_transcriber = None


class AudioTranscriber:
    def __init__(self, model_size: str = "base"):
        self.model = whisper.load_model(model_size)

    def _has_audio_stream(self, file_path: str) -> bool:
        """Verifica se o arquivo tem stream de áudio via ffprobe."""
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_streams", "-select_streams", "a", file_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)
            return len(data.get("streams", [])) > 0
        except Exception:
            return False

    def transcribe(self, audio_path: str) -> dict:
        # Retorna resultado vazio se não houver áudio
        if not self._has_audio_stream(audio_path):
            return {
                "audio": os.path.basename(audio_path),
                "transcribed_at": datetime.now().isoformat(),
                "language": "pt",
                "text": "",
                "segments": [],
                "warning": "Nenhuma trilha de áudio encontrada no arquivo.",
            }

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


def transcribe_audio(audio_path: str, model_size: str = "base") -> dict:
    global _default_transcriber
    if _default_transcriber is None:
        _default_transcriber = AudioTranscriber(model_size=model_size)
    return _default_transcriber.transcribe(audio_path)
