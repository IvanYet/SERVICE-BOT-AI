import pyaudio
import wave
import torch
import numpy as np
import time
import noisereduce as nr
import config  # <--- NEW IMPORT

class Recorder:
    def __init__(self):
        self.CHUNK = config.RECORDER_CHUNK # Replaces 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = config.MIC_CHANNELS # Replaces 1
        self.RATE = config.MIC_RATE # Replaces 16000
        self.p = pyaudio.PyAudio()
        
        print("[RECORDER] Loading Silero VAD...")
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                           model='silero_vad',
                                           force_reload=False,
                                           trust_repo=True)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        print("[RECORDER] Ready.")

    def record(self, filename="command.wav", max_duration=config.MAX_RECORDING_DURATION): # Replaces 10
        #open mic
        print(f"[RECORDER] Listening...")
        stream = self.p.open(format=self.FORMAT, channels=self.CHANNELS,
                             rate=self.RATE, input=True, frames_per_buffer=self.CHUNK)
        frames = []
        silence_start = None
        has_spoken = False
        start_time = time.time()

        while True:
            if time.time() - start_time > max_duration:
                break
            
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            frames.append(data)
            
            # VAD Check
            audio_int16 = np.frombuffer(data, np.int16)
            audio_float32 = audio_int16.astype(np.float32) / 32768.0
            voice_prob = self.model(torch.from_numpy(audio_float32), 16000).item()
            
            if voice_prob > config.VAD_THRESHOLD: # Replaces 0.5
                has_spoken = True
                silence_start = None
            else:
                if has_spoken and silence_start is None:
                    silence_start = time.time()
                elif has_spoken and silence_start:
                    if time.time() - silence_start > config.SILENCE_LIMIT: # Replaces 1.5
                        break

        stream.stop_stream()
        stream.close()

        if len(frames) > 0:
            raw_audio = b''.join(frames)
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)
            # 1st : noise reduction
            #logic : takes sample of bg noise and remove from the whole file
            noise_profile = audio_np[:config.NOISE_SAMPLE_SIZE] # Replaces 8000
            clean_audio = nr.reduce_noise(y=audio_np, sr=self.RATE, y_noise=noise_profile, prop_decrease=config.NOISE_REDUCTION_AMOUNT) # Replaces 0.8

            # 2nd layer: sound boost 
            #logic :increase the volume receive by the mic
            max_peak = np.max(np.abs(clean_audio))
            if max_peak > 0:
                boost_factor = config.NORMALIZATION_TARGET / max_peak # Replaces 30000
                boost_factor = min(boost_factor, config.MAX_BOOST) # Replaces 3.0
                clean_audio = (clean_audio * boost_factor).astype(np.int16)

            # Save
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(clean_audio.tobytes())
            wf.close()
            return filename
        else:
            return None