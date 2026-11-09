from faster_whisper import WhisperModel
import time
from robo_cup_robot.speech import speech_config as config

class Scribe:
    def __init__(self):
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device="cpu",
            compute_type="int8",
            local_files_only=True,
        ) # Use the already-downloaded model without contacting Hugging Face at startup.

    def listen(self, audio_file_path):
        started = time.perf_counter()
        prompt = config.SCRIBE_PROMPT # Replaces the hardcoded string
        segments, _ = self.model.transcribe(audio_file_path, beam_size=config.WHISPER_BEAM_SIZE, initial_prompt=prompt) # Replaces 5
        text = " ".join(segment.text.strip() for segment in segments).strip()
        print(f"[TIMING] Whisper transcription: {time.perf_counter() - started:.2f}s")
        return text
