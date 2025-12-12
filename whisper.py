from faster_whisper import WhisperModel
import config  # <--- NEW IMPORT

class Scribe:
    def __init__(self):
        self.model = WhisperModel(config.WHISPER_MODEL, device="cpu", compute_type="int8") # Replaces "tiny.en"

    def listen(self, audio_file_path):
        # convert audio to text
        prompt = config.SCRIBE_PROMPT # Replaces the hardcoded string
        segments, _ = self.model.transcribe(audio_file_path, beam_size=config.WHISPER_BEAM_SIZE, initial_prompt=prompt) # Replaces 5
        
        # format the text 
        text = ""
        for segment in segments:
            text += segment.text + " "
            
        return text.strip()