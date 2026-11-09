import pyaudio
import wave
import torch
import numpy as np
import time
from pathlib import Path
import noisereduce as nr
from robo_cup_robot.speech import speech_config as config

class Recorder:
    def __init__(self):
        self.CHUNK = config.RECORDER_CHUNK # Replaces 512
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = config.MIC_CHANNELS # Replaces 1
        self.RATE = config.MIC_RATE # Replaces 16000
        self.p = pyaudio.PyAudio()
        
        print("[RECORDER] Loading Silero VAD...")
        cached_repo = Path(torch.hub.get_dir()) / "snakers4_silero-vad_master"
        if (cached_repo / "hubconf.py").is_file():
            self.model, utils = torch.hub.load(
                repo_or_dir=str(cached_repo),
                model="silero_vad",
                source="local",
            )
            print(f"[RECORDER] Loaded Silero VAD from cache: {cached_repo}")
        else:
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
            )
        self.model.eval()
        torch.set_num_threads(1)
        (self.get_speech_timestamps, _, self.read_audio, _, _) = utils
        self._noise_profile = None
        print("[RECORDER] Ready.")

    def _open_input_stream(self):
        return self.p.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=config.MIC_DEVICE_INDEX,
            frames_per_buffer=self.CHUNK,
        )

    def _capture_noise_profile(self, stream):
        calibration_frames = []
        calibrated_samples = 0
        while calibrated_samples < config.NOISE_SAMPLE_SIZE:
            data = stream.read(self.CHUNK, exception_on_overflow=False)
            calibration_frames.append(data)
            calibrated_samples += len(data) // 2
        return np.frombuffer(b"".join(calibration_frames), dtype=np.int16)

    def calibrate(self):
        """Measure room noise before speaking a prompt, so the answer onset is kept."""
        print("[RECORDER] Quick noise calibration...")
        started = time.perf_counter()
        stream = self._open_input_stream()
        try:
            self._noise_profile = self._capture_noise_profile(stream)
        finally:
            stream.stop_stream()
            stream.close()
        print(f"[TIMING] Noise calibration: {time.perf_counter() - started:.2f}s")

    def record(self, filename="command.wav", max_duration=config.MAX_RECORDING_DURATION): # Replaces 10
        #open mic
        print(f"[RECORDER] Listening...")
        capture_started = time.perf_counter()
        stream = self._open_input_stream()
        frames = []
        noise_profile = self._noise_profile
        self._noise_profile = None
        if noise_profile is None:
            # Keep direct Recorder.record() callers working; the receptionist
            # pre-calibrates before prompts so this fallback doesn't clip speech.
            print("[RECORDER] Quick noise calibration...")
            noise_profile = self._capture_noise_profile(stream)
        print("[RECORDER] Listening for speech...")
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
            audio_float32 = audio_int16.astype(np.float32) * (config.MIC_INPUT_GAIN / 32768.0)
            audio_float32 = np.clip(audio_float32, -1.0, 1.0)
            with torch.inference_mode():
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
        print(f"[TIMING] Recording capture and VAD: {time.perf_counter() - capture_started:.2f}s")

        if len(frames) > 0:
            raw_audio = b''.join(frames)
            audio_np = np.frombuffer(raw_audio, dtype=np.int16)
            # 1st : noise reduction
            #logic : takes sample of bg noise and remove from the whole file
            cleanup_started = time.perf_counter()
            clean_audio = nr.reduce_noise(
                y=audio_np,
                sr=self.RATE,
                y_noise=noise_profile,
                prop_decrease=config.NOISE_REDUCTION_AMOUNT,
                stationary=True,
                n_fft=config.NOISE_REDUCTION_FFT,
                hop_length=config.NOISE_REDUCTION_HOP,
                n_jobs=1,
            )
            print(f"[TIMING] Noise reduction: {time.perf_counter() - cleanup_started:.2f}s")

            # 2nd layer: sound boost 
            #logic :increase the volume receive by the mic
            normalize_started = time.perf_counter()
            max_peak = np.max(np.abs(clean_audio))
            if max_peak > 0:
                boost_factor = config.NORMALIZATION_TARGET / max_peak # Replaces 30000
                boost_factor = min(boost_factor, config.MAX_BOOST) # Replaces 3.0
                clean_audio = (clean_audio * boost_factor).astype(np.int16)
            print(f"[TIMING] Audio normalization: {time.perf_counter() - normalize_started:.2f}s")

            # Save
            write_started = time.perf_counter()
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(self.p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(clean_audio.tobytes())
            wf.close()
            print(f"[TIMING] Write audio file: {time.perf_counter() - write_started:.2f}s")
            return filename
        else:
            return None
