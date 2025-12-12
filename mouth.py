import subprocess
import pyaudio  
import platform
import os
import json
import config # <--- NEW IMPORT

class Mouth :
    def __init__(self, model_path=config.VOICE_PATH): # Default from config
        self.model_path = model_path
        self.p = pyaudio.PyAudio()

        #dynamic , see what system u use
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if platform.system() == "Windows":
            self.piper_path = os.path.join(current_dir, "piper_bin", "windows", "piper.exe")
        else:
            self.piper_path = os.path.join(current_dir, "piper_bin", "linux", "piper")

        #dynamic, reads json file and get the sample rate , if want to change audio by chance
        config_path = f"{model_path}.json"
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                json_conf = json.load(f)
                self.rate = json_conf['audio']['sample_rate']

        #if dynamic cannot then hard code
        else:
            self.rate = 22050 # fixed for current voice
            print("[MOUTH] Warning: Config not found. Using default 22050Hz.")
    
    def speak(self,text):
        stream = self.p.open(
            format=pyaudio.paInt16,  
            channels=1,              
            rate=self.rate,              
            output=True
        )

        #setup , use piper path n model path 
        command = [
            self.piper_path,
            "--model", self.model_path,
            "--output_raw" 
        ]

        process = subprocess.Popen(
            command, 
            stdin=subprocess.PIPE, 
            stdout=subprocess.PIPE
        )

        process.stdin.write(text.encode('utf-8'))
        process.stdin.close()

        while True:
            data = process.stdout.read(4096)
            if not data:
                break
            stream.write(data)

        stream.stop_stream()
        stream.close()
        process.wait()