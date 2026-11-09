"""Shared settings for the RoboCup robot modules."""

from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent.parent
SPEECH_DIR = Path(__file__).resolve().parent

# --- HARDWARE ---
MIC_RATE = 16000                    #fixed
MIC_CHANNELS = 1                    #fixed
MIC_DEVICE_INDEX = None             # Set to the external mic's PyAudio input index
MIC_INPUT_GAIN = 2.0                 # Digital gain before wake-word and speech detection; raise carefully

# --- EAR (Wake Word) ---
EAR_CHUNK = 1280                    #fixed
WAKE_WORD_THRESHOLD = 0.25          # Lower threshold makes "Hey Jarvis" easier to trigger

# --- RECORDER (Listening) ---
RECORDER_CHUNK = 512                #fixed
MAX_RECORDING_DURATION = 10         # Stop after 10 seconds each prompt
SILENCE_LIMIT = 0.9                 # Keep brief pauses from cutting short a response
VAD_THRESHOLD = 0.5                 #fixed

# --- NOISE REDUCTION ---
NOISE_SAMPLE_SIZE = 3200            # First 0.2 seconds used as noise profile
NOISE_REDUCTION_AMOUNT = 0.6        # Lighter filtering reduces preprocessing time
NOISE_REDUCTION_FFT = 512           # Small FFT keeps noise filtering responsive
NOISE_REDUCTION_HOP = 128
NORMALIZATION_TARGET = 30000        # Target volume level (Int16 max is ~32000)
MAX_BOOST = 3.0                     # Don't boost volume more than 3x (prevents static explosion)

# --- SCRIBE (Hearing) ---
WHISPER_MODEL = "base.en"           # More accurate English speech recognition
WHISPER_BEAM_SIZE = 3               # Better name/short-answer recognition than greedy decoding
SCRIBE_PROMPT = (
    "My name is. Is that correct? Yes. No. Please spell your name. "
    "My interest is. My favorite drink is."
)

# --- BRAIN (Thinking) ---
OLLAMA_MODEL = "qwen2.5:1.5b"       # Faster local model with enough capacity for short conversation
OLLAMA_NUM_PREDICT = 24             # Keep spoken replies concise
RECEPTIONIST1_NUM_PREDICT = 48      # Name-intent JSON is short; cap unnecessary decoding
RECEPTIONIST1_LLM_FALLBACK = False  # Repeat unclear speech instead of waiting on a slow model
OLLAMA_NUM_CTX = 1024               # Enough context for short voice conversations
OLLAMA_NUM_THREAD = 8               # Match the i5-12450H's eight physical cores for CPU fallback
OLLAMA_RECENT_MESSAGES = 4          # Send the last two exchanges plus the system prompt
MEMORY_FILE = str(SPEECH_DIR / "memory.json")
MEMORY_LIMIT = 10                   # Keep last 10 turns of conversation (10 line memory can increase but if increase the higher the slower)
#another important one : ai prompt , u set how u wan your ai to be
SYSTEM_PROMPT = (
    "You are Jarvis. A service robot. "
    "You have a continuous memory. "
    "ALWAYS check the previous messages for context. "
    "If the user told you their name previously, use it. "
    "Answer in one concise sentence, ideally under 20 words."
)

# --- MOUTH (Speaking) ---
VOICE_PATH = str(SPEECH_DIR / "voices" / "en_US-hfc_female-medium.onnx")
