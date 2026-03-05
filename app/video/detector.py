from ultralytics import YOLO
import cv2

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    MEDIAPIPE_AVAILABLE = True
except AttributeError:
    MEDIAPIPE_AVAILABLE = False

CONTEXT_MODELS = {
    "cirurgia": "../models/surgery.pt",
    "consulta": "../models/consultation.pt",
    "fisioterapia": "../models/physio.pt",
    "triagem": "../models/violence.pt",
}


class VideoDetector:
    def __init__(self, context: str = "cirurgia"):
        model_path = CONTEXT_MODELS.get(context, "models/yolov8n.pt")
        self.model = YOLO(model_path)
        self.context = context
        self.pose = None
        if MEDIAPIPE_AVAILABLE and context in ["fisioterapia", "triagem", "consulta"]:
            self.pose = mp_pose.Pose(min_detection_confidence=0.5)

    def analyze_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []

        # Modelos classify retornam probs, não boxes
        if results.boxes is not None:
            for box in results.boxes:
                detections.append({
                    "class": results.names[int(box.cls)],
                    "confidence": float(box.conf),
                    "bbox": box.xyxy[0].tolist(),
                })
        elif results.probs is not None:
            # classify: retorna a classe com maior probabilidade
            top1 = int(results.probs.top1)
            detections.append({
                "class": results.names[top1],
                "confidence": float(results.probs.top1conf),
                "bbox": [],
            })

        pose_data = None
        if self.pose is not None:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(rgb)
            if pose_result.pose_landmarks:
                pose_data = [
                    [lm.x, lm.y, lm.z, lm.visibility]
                    for lm in pose_result.pose_landmarks.landmark
                ]

        return {"detections": detections, "pose": pose_data, "context": self.context}
