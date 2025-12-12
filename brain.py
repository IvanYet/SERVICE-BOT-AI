import ollama
import json  
import os    
import config 

class Brain:
    def __init__(self):
        self.model = config.OLLAMA_MODEL
        self.memory_file = config.MEMORY_FILE 
        
        # --- NEW: LOAD MEMORY ---
        # Checks if we have a save file. If yes, load it. If no, start fresh.
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.messages = json.load(f)
                print(f"[BRAIN] Loaded history from {self.memory_file}")
            except:
                self.messages = self._get_default_memory()
        else:
            self.messages = self._get_default_memory()
            
        print(f"[BRAIN] Online: {self.model}")

    # Helper to get the starting prompt
    def _get_default_memory(self):
        return [
            {
                'role': 'system',
                'content': config.SYSTEM_PROMPT
            }
        ]

    # --- NEW: SAVE MEMORY ---
    def save_memory(self):
        try:
            with open(self.memory_file, 'w') as f:
                json.dump(self.messages, f, indent=4)
        except Exception as e:
            print(f"[BRAIN ERROR] Save failed: {e}")

    def think(self, user_text):
        if not user_text:
            return None
 
        # 1. Add User Input to History
        self.messages.append({'role': 'user', 'content': user_text})

        print(f"[BRAIN] Thinking...")
        
        try:
            # 2. Send the WHOLE history to Ollama
            response = ollama.chat(model=self.model, messages=self.messages)
            reply = response['message']['content']
            
            # 3. Add AI Reply to History
            self.messages.append({'role': 'assistant', 'content': reply})
            
            # 4. Trim Memory (Keep last 10 turns)
            # Uses config limit now
            if len(self.messages) > (config.MEMORY_LIMIT + 1):
                self.messages = [self.messages[0]] + self.messages[-config.MEMORY_LIMIT:]
            
            # --- NEW: PERSISTENCE ---
            self.save_memory()
            
            return reply
            
        except Exception as e:
            print(f"[ERROR] {e}")
            return "Error."