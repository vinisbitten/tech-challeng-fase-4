from datetime import datetime


class FusionAnalyzer:
    def fuse(self, results: dict) -> dict:
        risk_scores = {"BAIXO": 1, "MÉDIO": 2, "ALTO": 3, "CRÍTICO": 4}
        all_risks = []
        all_recommendations = []
        alerts = []

        if "video" in results:
            video = results["video"]
            all_risks.append(video.get("risk_level", "BAIXO"))
            if video.get("alerts"):
                for a in video["alerts"]:
                    alerts.append({
                        "source": "video",
                        "type": a["type"],
                        "severity": a["severity"],
                        "timestamp": a.get("timestamp"),
                    })

        if "audio" in results:
            audio = results["audio"]
            emotion = audio.get("emotion_analysis", {})
            all_risks.append(emotion.get("risk_level", "BAIXO"))
            all_recommendations.extend(emotion.get("recommendations", []))
            for flag in emotion.get("risk_flags", {}).keys():
                alerts.append({
                    "source": "audio",
                    "type": flag,
                    "severity": "ALTO" if flag == "violencia" else "MÉDIO",
                })

        if "text" in results:
            text = results["text"]
            all_risks.append(text.get("risk_level", "BAIXO"))
            all_recommendations.extend(text.get("recommendations", []))
            for flag in text.get("risk_flags", {}).keys():
                alerts.append({
                    "source": "texto",
                    "type": flag,
                    "severity": "ALTO" if flag in ["violencia", "ginecologico"] else "MÉDIO",
                })

        max_risk = max(all_risks, key=lambda r: risk_scores.get(r, 0)) if all_risks else "BAIXO"

        return {
            "fused_at": datetime.now().isoformat(),
            "overall_risk_level": max_risk,
            "alerts": alerts,
            "recommendations": list(set(all_recommendations)),
            "notify_team": max_risk in ["ALTO", "CRÍTICO"],
            "modalities_analyzed": list(results.keys()),
        }
