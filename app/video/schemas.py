from pydantic import BaseModel
from typing import List, Dict, Any

class Alert(BaseModel):
    timestamp: float
    type: str
    severity: str
    confidence: float

class VideoAnalysisResponse(BaseModel):
    video: str
    analyzed_at: str
    duration_seconds: float
    total_detections: int
    detections_by_class: Dict[str, int]
    alerts: List[Alert]
    risk_level: str
    detections: List[Dict[str, Any]]
