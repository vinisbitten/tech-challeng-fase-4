from ultralytics import YOLO
import cv2, mediapipe as mp

mp_pose = mp.solutions.pose

CONTEXT_MODELS = {
    "cirurgia":     "models/yolov8_custom/surgery.pt",
    "consulta":     "models/yolov8_custom/consultation.pt",
    "fisioterapia": "models/yolov8_custom/physio.pt",
    "triagem":      "models/yolov8_custom/violence.pt",
}

class VideoDetector:
    def __init__(self, context: str = "cirurgia"):
        model_path = CONTEXT_MODELS.get(context, "models/yolov8n.pt")
        self.model = YOLO(model_path)
        self.context = context
        self.pose = mp_pose.Pose(min_detection_confidence=0.5)

    def analyze_frame(self, frame):
        results = self.model(frame, verbose=False)[0]
        detections = []
        for box in results.boxes:
            detections.append({
                "class": results.names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": box.xyxy[0].tolist()
            })

        # Pose para fisioterapia e triagem
        pose_data = None
        if self.context in ["fisioterapia", "triagem", "consulta"]:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_result = self.pose.process(rgb)
            if pose_result.pose_landmarks:
                pose_data = [[lm.x, lm.y, lm.z, lm.visibility]
                             for lm in pose_result.pose_landmarks.landmark]

        return {"detections": detections, "pose": pose_data, "context": self.context}
