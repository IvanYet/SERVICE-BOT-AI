import atexit
import json
import os
import platform
import subprocess
import tempfile
import threading
import time
import wave

import pyaudio

from robo_cup_robot.speech import speech_config as config


class Mouth:
    """Speak through a warmed Piper process and a reusable audio stream."""

    def __init__(self, model_path=config.VOICE_PATH):
        self.model_path = model_path
        self.p = pyaudio.PyAudio()
        self._stream = None
        self._lock = threading.Lock()
        self._process = None
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if platform.system() == "Windows":
            self.piper_path = os.path.join(current_dir, "piper_bin", "windows", "piper.exe")
        else:
            self.piper_path = os.path.join(current_dir, "piper_bin", "linux", "piper")

        config_path = f"{model_path}.json"
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as stream:
                self.rate = json.load(stream)["audio"]["sample_rate"]
        else:
            self.rate = 22050
            print("[MOUTH] Warning: Config not found. Using default 22050Hz.")

        # Load the voice once during startup rather than on every response.
        self._start_piper()
        atexit.register(self.close)

    def _start_piper(self):
        if self._process is not None and self._process.poll() is None:
            return
        self._process = subprocess.Popen(
            [self.piper_path, "--model", self.model_path, "--json-input", "--quiet"],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _output_stream(self):
        if self._stream is None:
            self._stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.rate,
                output=True,
            )
        return self._stream

    def speak(self, text, length_scale=1.0):
        self.speak_segments([(text, length_scale)])

    def speak_segments(self, segments):
        """Join phrase segments so Piper synthesizes the utterance only once."""
        items = [(str(text).strip(), float(scale)) for text, scale in segments if str(text).strip()]
        if not items:
            return
        # The bundled Piper CLI accepts a fixed synthesis rate per process.
        # Keep one voice process warm; punctuation still separates spelled names.
        text = " ".join(item[0] for item in items)
        with self._lock:
            self._start_piper()
            fd, wav_path = tempfile.mkstemp(prefix="robocup-tts-", suffix=".wav")
            os.close(fd)
            os.unlink(wav_path)
            synthesis_started = time.perf_counter()
            try:
                self._process.stdin.write(
                    json.dumps({"text": text, "output_file": wav_path}, ensure_ascii=False) + "\n"
                )
                self._process.stdin.flush()
                self._wait_for_audio(wav_path, timeout=max(30.0, len(text) * 0.5))
                synthesis_elapsed = time.perf_counter() - synthesis_started

                playback_started = time.perf_counter()
                stream = self._output_stream()
                with wave.open(wav_path, "rb") as audio_file:
                    audio_duration = audio_file.getnframes() / audio_file.getframerate()
                    while True:
                        audio_chunk = audio_file.readframes(2048)
                        if not audio_chunk:
                            break
                        stream.write(audio_chunk)
                print(
                    f"[TIMING] Piper synthesis: {synthesis_elapsed:.2f}s; "
                    f"audio duration: {audio_duration:.2f}s; "
                    f"playback: {time.perf_counter() - playback_started:.2f}s"
                )
            except Exception:
                self._restart_piper()
                raise
            finally:
                try:
                    os.unlink(wav_path)
                except FileNotFoundError:
                    pass

    def _wait_for_audio(self, path, timeout):
        deadline = time.monotonic() + timeout
        previous_size = -1
        stable_checks = 0
        while time.monotonic() < deadline:
            if self._process.poll() is not None:
                raise RuntimeError(f"Piper exited with status {self._process.returncode}")
            try:
                size = os.path.getsize(path)
            except FileNotFoundError:
                size = 0
            if size > 44 and size == previous_size:
                stable_checks += 1
                if stable_checks >= 3:
                    return
            else:
                stable_checks = 0
            previous_size = size
            time.sleep(0.02)
        raise TimeoutError("Piper did not finish generating speech audio in time")

    def _restart_piper(self):
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        self._start_piper()

    def close(self):
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                try:
                    self._process.stdin.close()
                    self._process.wait(timeout=2.0)
                except (BrokenPipeError, subprocess.TimeoutExpired):
                    self._process.terminate()
            self._process = None
            if self._stream is not None:
                self._stream.stop_stream()
                self._stream.close()
                self._stream = None
            if self.p is not None:
                self.p.terminate()
                self.p = None
