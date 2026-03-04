import os
from openai import OpenAI
from datetime import datetime

RISK_KEYWORDS = {
    "depressao": ["triste", "sem esperanca", "nao quero mais", "cansada", "choro", "vazio", "desanimo"],
    "ansiedade": ["nervosa", "ansiosa", "preocupada", "medo", "panico", "tensao", "angustia"],
    "violencia": ["ele me bate", "tenho medo dele", "nao posso sair", "me machuca", "ameaca"],
    "pos_parto": ["nao consigo cuidar", "nao sinto amor", "arrependida", "exausta", "nao durmo"],
}


class EmotionAnalyzer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key and api_key != "sua_chave_aqui" else None

    def analyze(self, audio_path: str, transcription_text: str) -> dict:
        keyword_flags = self._detect_keywords(transcription_text)
        risk_level = self._assess_risk(keyword_flags)

        llm_analysis = None
        if self.client:
            llm_analysis = self._analyze_with_llm(transcription_text)

        return {
            "emotion_analysis": {
                "analyzed_at": datetime.now().isoformat(),
                "risk_flags": keyword_flags,
                "risk_level": risk_level,
                "llm_analysis": llm_analysis,
                "recommendations": self._get_recommendations(keyword_flags),
            }
        }

    def _detect_keywords(self, text: str) -> dict:
        text_lower = text.lower()
        flags = {}
        for category, keywords in RISK_KEYWORDS.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                flags[category] = found
        return flags

    def _assess_risk(self, flags: dict) -> str:
        if not flags:
            return "BAIXO"
        if "violencia" in flags:
            return "CRÍTICO"
        if len(flags) >= 2:
            return "ALTO"
        return "MÉDIO"

    def _analyze_with_llm(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente clínico especializado em saúde da mulher. "
                            "Analise o texto da consulta e identifique sinais de risco psicológico, "
                            "depressão pós-parto, ansiedade ou violência doméstica. "
                            "Seja objetivo e clínico. Responda em português."
                        )
                    },
                    {"role": "user", "content": f"Analise este texto de consulta:\n\n{text}"}
                ],
                max_tokens=500,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Análise LLM indisponível: {str(e)}"

    def _get_recommendations(self, flags: dict) -> list:
        recs = []
        if "violencia" in flags:
            recs.append("URGENTE: Acionar protocolo de proteção à mulher vítima de violência.")
        if "depressao" in flags or "pos_parto" in flags:
            recs.append("Encaminhar para avaliação psiquiátrica imediata.")
        if "ansiedade" in flags:
            recs.append("Considerar acompanhamento psicológico e avaliação de ansiedade gestacional.")
        if not recs:
            recs.append("Nenhuma ação imediata necessária. Manter acompanhamento de rotina.")
        return recs
