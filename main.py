import os
import config # <--- NEW IMPORT
from ear import Ear
from mouth import Mouth
from whisper import Scribe  
from recorder import Recorder
from brain import Brain  

def main():
    print("--- SYSTEM STARTUP ---")
    
    # Initialize (Defaults are now loaded from config.py)
    ear = Ear()
    mouth = Mouth() 
    scribe = Scribe()
    recorder = Recorder()
    brain = Brain()

    mouth.speak("System online.")
    print("--- READY ---")

    while True:
        
        # A. Activate
        if ear.wait_for_wake_word():
            mouth.speak("Yes?")
            
            # B. Record (Duration and logic controlled by config)
            filename = recorder.record("command.wav")
            
            if filename:
                # C. Translate
                user_text = scribe.listen(filename)
                print(f"[USER]: {user_text}")
                
                # D. Brain
                if "stop" in user_text.lower():
                    mouth.speak("Stopping motors.")
                else:
                    ai_reply = brain.think(user_text)
                    print(f"[JARVIS]: {ai_reply}")
                    mouth.speak(ai_reply)             
                
                # E. Cleanup
                if os.path.exists(filename):
                    os.remove(filename)
            
            print("--- WAITING ---")

if __name__ == "__main__":
    main()