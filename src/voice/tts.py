import re
import asyncio
from pathlib import Path
from typing import Optional
from src.config import DEFAULT_TTS_VOICE, AUDIO_OUTPUT_DIR

class VoiceSynthesisNode:
    """
    Decoupled Voice Output: Edge-TTS Neural Speech Synthesis.
    Converts grounded LLM responses into high-fidelity natural audio streams.
    """

    def __init__(self, voice: str = DEFAULT_TTS_VOICE):
        self.voice = voice

    @staticmethod
    def clean_text_for_speech(text: str) -> str:
        """
        Removes markdown citations, asterisks, brackets, and URLs so the voice output sounds natural.
        """
        # Remove citation brackets like [Page 1], [1]
        cleaned = re.sub(r'\[Page\s*\d+\]', '', text, flags=re.IGNORECASE)
        cleaned = re.sub(r'\[\d+\]', '', cleaned)
        # Remove markdown bold/italics
        cleaned = re.sub(r'[*_#`]', '', cleaned)
        # Remove markdown links
        cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        return cleaned

    async def _synthesize_async(self, text: str, output_path: Path) -> Path:
        import edge_tts
        cleaned_text = self.clean_text_for_speech(text)
        if not cleaned_text:
            cleaned_text = "No response generated."

        communicate = edge_tts.Communicate(cleaned_text, self.voice)
        await communicate.save(str(output_path))
        return output_path

    def generate_speech(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        """
        Generates an MP3 audio file from the input text synchronously.
        """
        try:
            if not filename:
                import uuid
                filename = f"response_{uuid.uuid4().hex[:8]}.mp3"

            out_path = AUDIO_OUTPUT_DIR / filename

            # Run in asyncio event loop
            try:
                asyncio.run(self._synthesize_async(text, out_path))
            except RuntimeError:
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    fut = pool.submit(asyncio.run, self._synthesize_async(text, out_path))
                    fut.result(timeout=12)

            if out_path.exists():
                return str(out_path)
            return None
        except Exception as e:
            print(f"[VoiceSynthesisNode] Speech generation warning: {e}")
            return None
