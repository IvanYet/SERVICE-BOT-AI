import pyaudio
import numpy as np
from openwakeword.model import Model
from robo_cup_robot.speech import speech_config as config

class Ear:
    def __init__(self):
        self.model = Model(inference_framework="onnx")
        self.p = pyaudio.PyAudio()
        self.CHUNK = config.EAR_CHUNK # Replaces 1280
        self.RATE = config.MIC_RATE   # Replaces 16000

    def wait_for_wake_word(self):
        print("Say 'Hey Jarvis'...")
        
        #open mic
        stream = self.p.open(
            format=pyaudio.paInt16, 
            channels=config.MIC_CHANNELS, # Replaces 1
            rate=self.RATE, 
            input=True, 
            input_device_index=config.MIC_DEVICE_INDEX,
            frames_per_buffer=self.CHUNK
        )

        try:
            while True:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                audio_int = np.frombuffer(data, dtype=np.int16)
                # Apply modest digital gain before wake-word detection. Clip to int16
                # so louder samples cannot wrap around and distort unpredictably.
                if config.MIC_INPUT_GAIN != 1.0:
                    audio_int = np.clip(
                        audio_int.astype(np.float32) * config.MIC_INPUT_GAIN,
                        -32768, 32767
                    ).astype(np.int16)
                prediction = self.model.predict(audio_int)
                
                if prediction['hey_jarvis'] > config.WAKE_WORD_THRESHOLD: # Replaces 0.5
                    self.model.reset()
                    stream.stop_stream()
                    stream.close()
                    return True
                    
        except Exception as e:
            print(f"Ear Error: {e}")
            stream.stop_stream()
            stream.close()
            return False
