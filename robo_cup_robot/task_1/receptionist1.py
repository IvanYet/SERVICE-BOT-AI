"""Task-scoped dialogue for the first Receptionist interaction."""

import json
import re
import time

import ollama

from robo_cup_robot.speech import speech_config as config


class Receptionist1:
    """Handle one guest's greeting and name confirmation.

    Each instance starts with fresh task-only conversation history. It does not
    read or write the general assistant's persistent ``memory.json``.
    """

    def __init__(self):
        self.messages = [
            {
                "role": "system",
                "content": (
                    "You interpret guest speech for the name-confirmation step of a "
                    "RoboCup@Home Receptionist task. Do not write dialogue or decide "
                    "when the task is complete. Follow the phase included with each "
                    "transcript. In phase await_name, extract only the guest's name, "
                    "ignoring unrelated trailing words. In phase confirm_name, classify "
                    "the response as confirm, deny, correction, or unclear. In phase "
                    "spell_name, interpret the letters the guest spells as a name. "
                    "Preserve unusual names and spelling; never autocorrect to a common "
                    "name. Return only JSON with keys intent and value. intent must be "
                    "name, confirm, deny, correction, spelling, or unclear. value must "
                    "be the extracted name or null. Do not include markdown or extra keys."
                ),
            }
        ]
        self.complete = False
        self.confirmed_name = None
        self.candidate_name = None
        self.phase = "await_name"

    def start(self):
        """Return the fixed opening for this phase."""
        opening = (
            "Hello, guest. Please look at me and smile. Please wait until I finish "
            "speaking before you answer. What is your name?"
        )
        self.messages.append({"role": "assistant", "content": opening})
        return opening

    def respond(self, guest_text):
        """Interpret one transcript and enforce the name-confirmation sequence."""
        if self.complete or not guest_text.strip():
            return None

        clean_text = guest_text.strip()
        self.messages.append({"role": "user", "content": clean_text})

        if self.phase == "await_name":
            name = self._extract_name(clean_text)
            if name:
                self.candidate_name = name
                self.phase = "confirm_name"
                return self._remember_reply(self._confirmation_question())

        if self.phase in ("confirm_name", "spell_name"):
            corrected_name = self._extract_spelled_name(clean_text)
            if corrected_name:
                self.candidate_name = corrected_name
                self.phase = "confirm_name"
                return self._remember_reply(self._confirmation_question())

            corrected_name = self._extract_correction_name(clean_text)
            if corrected_name:
                self.candidate_name = corrected_name
                self.phase = "confirm_name"
                return self._remember_reply(self._confirmation_question())

        if self.phase == "confirm_name":
            answer = self._yes_no(clean_text)
            if answer == "yes":
                self.complete = True
                self.confirmed_name = self.candidate_name
                return self._remember_reply(f"Thank you, {self.confirmed_name}.")
            if answer == "no":
                self.phase = "spell_name"
                return self._remember_reply(
                    "Please spell your full name for me, one letter at a time."
                )

        # Use the task-scoped model context only when deterministic parsing is
        # insufficient, such as a longer spoken correction.
        if not config.RECEPTIONIST1_LLM_FALLBACK:
            if self.phase == "await_name":
                return self._remember_reply("Sorry, I didn't catch your name. What is your name?")
            if self.phase == "spell_name":
                return self._remember_reply("Please spell your name one letter at a time.")
            return self._remember_reply(self._confirmation_question())

        self.messages[-1]["content"] = f"Phase: {self.phase}. Guest transcript: {clean_text}"
        llm_started = time.perf_counter()
        response = ollama.chat(
            model=config.OLLAMA_MODEL,
            messages=self.messages,
            format="json",
            options={
                "num_predict": config.RECEPTIONIST1_NUM_PREDICT,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_thread": config.OLLAMA_NUM_THREAD,
            },
            keep_alive="15m",
        )
        print(f"[TIMING] Receptionist name interpretation: {time.perf_counter() - llm_started:.2f}s")
        content = response["message"]["content"]
        try:
            result = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return self._remember_reply("Sorry, I had trouble understanding that. Could you repeat that?")

        if not isinstance(result, dict):
            return self._remember_reply("Sorry, I had trouble understanding that. Could you repeat that?")

        intent = result.get("intent")
        value = result.get("value")
        value = value.strip(" \t\r\n.,!?\"'") if isinstance(value, str) else ""

        if self.phase == "confirm_name":
            if intent == "confirm":
                self.complete = True
                self.confirmed_name = self.candidate_name
                return self._remember_reply(f"Thank you, {self.confirmed_name}.")
            if intent == "deny":
                self.phase = "spell_name"
                return self._remember_reply("Please spell your full name for me, one letter at a time.")
            if intent in ("name", "correction", "spelling") and value:
                self.candidate_name = value
                return self._remember_reply(self._confirmation_question())
            return self._remember_reply(self._confirmation_question())

        if self.phase == "spell_name":
            if intent in ("name", "correction", "spelling") and value:
                self.candidate_name = value
                self.phase = "confirm_name"
                return self._remember_reply(self._confirmation_question())
            return self._remember_reply("Please spell your name one letter at a time.")

        if self.phase == "await_name":
            if intent in ("name", "spelling", "correction") and value:
                self.candidate_name = value
                self.phase = "confirm_name"
                return self._remember_reply(self._confirmation_question())
            return self._remember_reply("Sorry, I didn't catch your name. What is your name?")

        return self._remember_reply("Sorry, could you repeat your name?")

    def _confirmation_question(self):
        """Build a deterministic spoken spelling from the extracted name."""
        spelled_words = []
        for word in self.candidate_name.split():
            letters = "-".join(character.upper() for character in word if character.isalnum())
            if letters:
                spelled_words.append(letters)
        spelled_name = " ".join(spelled_words)
        return f"I heard {self.candidate_name}. Is that spelled {spelled_name}?"

    @staticmethod
    def _extract_name(guest_text):
        """Extract a name from a natural answer without a fixed name list."""
        text = guest_text.strip().strip(" \t\r\n.,!?\"'")
        name_intro = re.search(
            r"\b(?:my name is|i am|i'm|this is|it is|it's)\s+(.+)$",
            text,
            flags=re.IGNORECASE,
        )
        if name_intro:
            text = name_intro.group(1)

        # Ignore a trailing clause if speech recognition includes more than the name.
        text = re.split(r"\b(?:and|but|because|i|we|you|he|she)\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
        text = text.strip(" \t\r\n.,!?\"'")
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", text)
        if not words or len(words) > 4:
            return None
        return " ".join(words)

    def _extract_spelled_name(self, guest_text):
        """Read explicit letter sequences and preserve the current name's word breaks."""
        match = re.search(r"\bspell(?:ed|ing)?\s+(.+)$", guest_text, flags=re.IGNORECASE)
        spelling_text = match.group(1) if match else guest_text
        spelling_text = re.split(
            r"\b(?:sorry|can i|could i|may i|please|again)\b",
            spelling_text,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        sequences = re.findall(r"(?:[A-Za-z]-){1,}[A-Za-z]", spelling_text)
        if sequences:
            words = [re.sub(r"[^A-Za-z]", "", sequence).capitalize() for sequence in sequences]
            if len(words) > 1:
                return " ".join(words)
            letters = words[0]
        else:
            tokens = re.findall(r"[A-Za-z]+", spelling_text)
            if not tokens:
                return None
            if all(len(token) == 1 for token in tokens):
                letters = "".join(tokens).capitalize()
            elif match and len(tokens) <= 4:
                return " ".join(token.capitalize() for token in tokens)
            else:
                return None

        old_word_lengths = [len(re.sub(r"[^A-Za-z]", "", word)) for word in self.candidate_name.split()]
        if len(old_word_lengths) > 1 and sum(old_word_lengths) == len(letters):
            parts = []
            offset = 0
            for word_length in old_word_lengths:
                parts.append(letters[offset:offset + word_length])
                offset += word_length
            return " ".join(part.capitalize() for part in parts)
        return letters.capitalize()

    @staticmethod
    def _extract_correction_name(guest_text):
        """Extract ordinary spoken corrections such as “it's supposed to be Usyk.”"""
        correction = re.search(
            r"\b(?:it is supposed to be|it's supposed to be|it should be|"
            r"the correct (?:name|spelling) is|the name is|my name is|i meant|"
            r"it is about|it's about|it is|it's)\s+(.+)$",
            guest_text,
            flags=re.IGNORECASE,
        )
        if not correction:
            return None

        value = re.split(
            r"\b(?:sorry|can i|could i|may i|again|but)\b",
            correction.group(1),
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0]
        words = re.findall(r"[A-Za-z][A-Za-z'’-]*", value.strip())
        if not words or len(words) > 4:
            return None
        return " ".join(word.capitalize() for word in words)

    @staticmethod
    def _yes_no(guest_text):
        normalized = re.sub(r"[^a-z' ]", " ", guest_text.lower())
        normalized = " ".join(normalized.split())
        if re.match(r"^(yes|yeah|yep|yup|correct|right|that's right|that is right)\b", normalized):
            return "yes"
        if re.match(r"^(no|nope|incorrect|wrong|that's not right|that is not right)\b", normalized):
            return "no"
        return None

    def _remember_reply(self, reply):
        self.messages.append({"role": "assistant", "content": reply})
        return reply
