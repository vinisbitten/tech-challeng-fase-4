import os
import azure.cognitiveservices.speech as speechsdk
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
from datetime import datetime


class AzureSpeechService:
    def __init__(self):
        self.key = os.getenv("AZURE_SPEECH_KEY")
        self.region = os.getenv("AZURE_SPEECH_REGION", "eastus")

    def transcribe_audio(self, audio_path: str) -> dict:
        if not self.key or self.key == "sua_chave_aqui":
            return {"error": "Azure Speech Key não configurada.", "text": ""}

        speech_config = speechsdk.SpeechConfig(subscription=self.key, region=self.region)
        speech_config.speech_recognition_language = "pt-BR"
        audio_config = speechsdk.audio.AudioConfig(filename=audio_path)
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)

        result = recognizer.recognize_once_async().get()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return {
                "provider": "azure",
                "transcribed_at": datetime.now().isoformat(),
                "text": result.text,
            }
        return {"provider": "azure", "error": str(result.reason), "text": ""}


class AzureLanguageService:
    def __init__(self):
        self.key = os.getenv("AZURE_LANGUAGE_KEY")
        self.endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT")

    def analyze_sentiment(self, text: str) -> dict:
        if not self.key or self.key == "sua_chave_aqui":
            return {"error": "Azure Language Key não configurada."}

        client = TextAnalyticsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

        response = client.analyze_sentiment([text], language="pt")[0]

        return {
            "provider": "azure",
            "sentiment": response.sentiment,
            "scores": {
                "positive": round(response.confidence_scores.positive, 3),
                "neutral": round(response.confidence_scores.neutral, 3),
                "negative": round(response.confidence_scores.negative, 3),
            }
        }

    def extract_key_phrases(self, text: str) -> dict:
        if not self.key or self.key == "sua_chave_aqui":
            return {"error": "Azure Language Key não configurada."}

        client = TextAnalyticsClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )

        response = client.extract_key_phrases([text], language="pt")[0]
        return {
            "provider": "azure",
            "key_phrases": list(response.key_phrases),
        }
