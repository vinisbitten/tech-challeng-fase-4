from ultralytics import YOLO
import cv2
from pathlib import Path
from datetime import datetime

ALERT_CLASSES = {"sangramento_anomalo", "sinal_desconforto"}


class VideoAnalyzer:
    def __init__(self, model_path: str = "yolov8n.pt"):
        self.model = YOLO(model_path)
        self.detections = []
        self.alerts = []

    def analyze_video(self, video_path: str, output_path: str = None) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        self.detections = []
        self.alerts = []
        frame_count = 0

        writer = None
        if output_path:
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (w, h)
            )

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if frame_count % max(int(fps or 1), 1) != 0:
                if writer:
                    writer.write(frame)
                continue

            results = self.model(frame, verbose=False)
            timestamp = frame_count / fps

            for result in results:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    cls_name = self.model.names.get(cls_id, f"classe_{cls_id}")

                    self.detections.append({
                        "timestamp": round(timestamp, 2),
                        "class": cls_name,
                        "confidence": round(confidence, 3),
                        "bbox": box.xyxy[0].tolist(),
                    })

                    if cls_name in ALERT_CLASSES and confidence > 0.6:
                        self.alerts.append({
                            "timestamp": round(timestamp, 2),
                            "type": cls_name,
                            "severity": "ALTO" if confidence > 0.8 else "MÉDIO",
                            "confidence": round(confidence, 3),
                        })

                if writer:
                    writer.write(result.plot())

        cap.release()
        if writer:
            writer.release()

        return self._build_report(video_path, duration)

    def _build_report(self, video_path: str, duration: float) -> dict:
        class_counts = {}
        for d in self.detections:
            class_counts[d["class"]] = class_counts.get(d["class"], 0) + 1

        return {
            "video": Path(video_path).name,
            "analyzed_at": datetime.now().isoformat(),
            "duration_seconds": round(duration, 2),
            "total_detections": len(self.detections),
            "detections_by_class": class_counts,
            "alerts": self.alerts,
            "risk_level": self._assess_risk(),
            "detections": self.detections,
        }

    def _assess_risk(self) -> str:
        if not self.alerts:
            return "BAIXO"
        high = sum(1 for a in self.alerts if a["severity"] == "ALTO")
        if high >= 3:
            return "CRÍTICO"
        elif high >= 1:
            return "ALTO"
        return "MÉDIO"
