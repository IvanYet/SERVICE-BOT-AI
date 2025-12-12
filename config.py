# config.py

# --- HARDWARE ---
MIC_RATE = 16000                    #fixed
MIC_CHANNELS = 1                    #fixed

# --- EAR (Wake Word) ---
EAR_CHUNK = 1280                    #fixed
WAKE_WORD_THRESHOLD = 0.5           #mic for activate senstivity 0-1

# --- RECORDER (Listening) ---
RECORDER_CHUNK = 512                #fixed
MAX_RECORDING_DURATION = 10         # Stop after 10 seconds each prompt
SILENCE_LIMIT = 1.5                 # Stop recording if user is silent for 1.5s
VAD_THRESHOLD = 0.5                 #fixed

# --- NOISE REDUCTION ---
NOISE_SAMPLE_SIZE = 8000            # First 0.5 seconds used as noise profile
NOISE_REDUCTION_AMOUNT = 0.8        # 0.8 = Remove 80% of noise
NORMALIZATION_TARGET = 30000        # Target volume level (Int16 max is ~32000)
MAX_BOOST = 3.0                     # Don't boost volume more than 3x (prevents static explosion)

# --- SCRIBE (Hearing) ---
WHISPER_MODEL = "tiny.en"           # txt-to-speech
WHISPER_BEAM_SIZE = 5               # 1 = Fast, 5 = Accurate (Better for accents)
SCRIBE_PROMPT = "Jarvis, Tony, Robot, Stop, Yes, No, Lights, Kitchen." #!!! important!! this is keywords

# --- BRAIN (Thinking) ---
OLLAMA_MODEL = "qwen2.5:3b"         #ai model
MEMORY_FILE = "memory.json"         #fixed
MEMORY_LIMIT = 10                   # Keep last 10 turns of conversation (10 line memory can increase but if increase the higher the slower)
#another important one : ai prompt , u set how u wan your ai to be
SYSTEM_PROMPT = (
    "You are Jarvis. A service robot. "
    "You have a continuous memory. "
    "ALWAYS check the previous messages for context. "
    "If the user told you their name previously, use it. "
    "Keep answers short."
)

# --- MOUTH (Speaking) ---
VOICE_PATH = "voices/en_US-hfc_female-medium.onnx" #fixed