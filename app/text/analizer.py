import os
from openai import OpenAI
from datetime import datetime

CLINICAL_KEYWORDS = {
    "risco_gestacional": ["pressao alta", "pre-eclampsia", "diabetes gestacional", "sangramento", "contracao precoce"],
    "saude_mental": ["depressao", "ansiedade", "insonia", "choro", "tristeza", "humor deprimido"],
    "ginecologico": ["mioma", "endometriose", "cisto", "cancer", "carcinoma", "displasia"],
    "violencia": ["abuso", "agressao", "violencia", "medo do parceiro", "lesao"],
}


class TextAnalyzer:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=api_key) if api_key and api_key != "sua_chave_aqui" else None

    def analyze(self, text: str) -> dict:
        keyword_flags = self._detect_keywords(text)
        risk_level = self._assess_risk(keyword_flags)

        llm_analysis = None
        if self.client:
            llm_analysis = self._analyze_with_llm(text)

        return {
            "analyzed_at": datetime.now().isoformat(),
            "text_length": len(text),
            "risk_flags": keyword_flags,
            "risk_level": risk_level,
            "llm_analysis": llm_analysis,
            "recommendations": self._get_recommendations(keyword_flags),
        }

    def _detect_keywords(self, text: str) -> dict:
        text_lower = text.lower()
        flags = {}
        for category, keywords in CLINICAL_KEYWORDS.items():
            found = [kw for kw in keywords if kw in text_lower]
            if found:
                flags[category] = found
        return flags

    def _assess_risk(self, flags: dict) -> str:
        if not flags:
            return "BAIXO"
        if "violencia" in flags or "ginecologico" in flags:
            return "ALTO"
        if len(flags) >= 2:
            return "MÉDIO"
        return "BAIXO"

    def _analyze_with_llm(self, text: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Você é um assistente médico especializado em saúde da mulher. "
                            "Analise o laudo ou texto clínico fornecido, identifique achados relevantes, "
                            "riscos e sugira condutas. Responda em português de forma objetiva."
                        )
                    },
                    {"role": "user", "content": f"Analise este laudo/texto clínico:\n\n{text}"}
                ],
                max_tokens=600,
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Análise LLM indisponível: {str(e)}"

    def _get_recommendations(self, flags: dict) -> list:
        recs = []
        if "violencia" in flags:
            recs.append("URGENTE: Acionar protocolo de proteção à mulher.")
        if "ginecologico" in flags:
            recs.append("Encaminhar para oncologista ou ginecologista especialista.")
        if "risco_gestacional" in flags:
            recs.append("Monitoramento intensivo da gestação. Considerar internação preventiva.")
        if "saude_mental" in flags:
            recs.append("Encaminhar para avaliação psiquiátrica.")
        if not recs:
            recs.append("Nenhuma ação imediata necessária.")
        return recs
