import os
from pathlib import Path
from typing import Optional
from src.config import GEMINI_API_KEY, DEFAULT_GEMINI_MODEL

class SpeechToTextNode:
    """
    Decoupled Voice Input: Speech-to-Text Transcriber.
    Supports local Whisper, Gemini Multimodal Audio transcription, and audio file processing.
    """

    def __init__(self, api_key: str = GEMINI_API_KEY):
        self.api_key = api_key

    def transcribe_audio_file(self, audio_path: str) -> str:
        """
        Transcribes a spoken audio file (.wav, .mp3, .ogg, .webm) into clean text.
        """
        path = Path(audio_path)
        if not path.exists():
            return ""

        # Strategy 1: Try local faster-whisper if available
        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("base.en", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(str(path), beam_size=5)
            transcription = " ".join([segment.text for segment in segments]).strip()
            if transcription:
                return transcription
        except Exception:
            pass

        # Strategy 2: Gemini Audio API (Native Multimodal)
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
                
                # Upload audio or pass bytes
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()

                # Determine mime type
                ext = path.suffix.lower()
                mime = "audio/wav"
                if ext == ".mp3":
                    mime = "audio/mp3"
                elif ext == ".ogg":
                    mime = "audio/ogg"
                elif ext == ".webm":
                    mime = "audio/webm"

                model = genai.GenerativeModel(DEFAULT_GEMINI_MODEL)
                response = model.generate_content([
                    "Please transcribe this spoken audio recording exactly word-for-word. Do not summarize or add commentary. Return only the transcription text.",
                    {"mime_type": mime, "data": audio_bytes}
                ])
                if response and response.text:
                    return response.text.strip()
            except Exception as e:
                print(f"[SpeechToTextNode] Gemini audio transcription error: {e}")

        # Strategy 3: Standard SpeechRecognition library fallback
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            with sr.AudioFile(str(path)) as source:
                audio_data = recognizer.record(source)
                return recognizer.recognize_google(audio_data)
        except Exception:
            pass

        return ""
