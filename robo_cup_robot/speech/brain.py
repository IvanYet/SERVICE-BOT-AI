import ollama
import json  
import os    
import time
from robo_cup_robot.speech import speech_config as config

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

        # Keep runtime prompt changes effective for existing memory files.
        if self.messages and self.messages[0].get('role') == 'system':
            self.messages[0]['content'] = config.SYSTEM_PROMPT
            
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
        request_started = time.perf_counter()
        
        try:
            # Keep stored memory intact, but send only recent exchanges to
            # reduce prompt work and avoid growing inference time over time.
            recent_messages = self.messages[1:][-config.OLLAMA_RECENT_MESSAGES:]
            request_messages = [self.messages[0]] + recent_messages
            response = ollama.chat(
                model=self.model,
                messages=request_messages,
                options={
                    "num_predict": config.OLLAMA_NUM_PREDICT,
                    "num_ctx": config.OLLAMA_NUM_CTX,
                    "num_thread": config.OLLAMA_NUM_THREAD,
                },
                keep_alive="15m",
            )
            request_elapsed = time.perf_counter() - request_started
            print(f"[BRAIN] Ollama request completed in {request_elapsed:.2f}s")
            # Ollama reports these durations in nanoseconds. Comparing them
            # with the client timer distinguishes model work from client wait.
            def metric(name):
                if isinstance(response, dict):
                    return response.get(name)
                return getattr(response, name, None)

            for name in ("load_duration", "prompt_eval_duration", "eval_duration"):
                value = metric(name)
                if value is not None:
                    print(f"[BRAIN] Ollama {name}: {value / 1_000_000_000:.2f}s")
            print(
                f"[BRAIN] Ollama tokens: prompt={metric('prompt_eval_count')}, "
                f"generated={metric('eval_count')}"
            )
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
            request_elapsed = time.perf_counter() - request_started
            print(f"[BRAIN] Ollama request failed after {request_elapsed:.2f}s")
            print(f"[ERROR] {e}")
            return "Error."
