import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from robo_cup_robot.speech.ear import Ear
from robo_cup_robot.speech.mouth import Mouth
from robo_cup_robot.speech.whisper import Scribe
from robo_cup_robot.speech.recorder import Recorder
from robo_cup_robot.task_1.receptionist1 import Receptionist1
from robo_cup_robot.vision.receptionist_monitor import ReceptionistMonitor
from robo_cup_robot.vision.drink_finder import find_favorite_drink

GUEST_RECORDS_FILE = Path(__file__).resolve().parents[1] / "task_1" / "guest_records.json"
DOOR_POSE = {"x": -1.2464, "y": -5.5788, "yaw": -1.5734}
TABLE_POSE = {"x": -1.22878, "y": -4.03087, "yaw": 0.12896}
TABLE_BACKWARD_POSE = {
    "x": TABLE_POSE["x"],
    "y": TABLE_POSE["y"],
    "yaw": TABLE_POSE["yaw"] - 3.141592653589793,
}
SEATING_AREA_POSE = {"x": -1.1901, "y": -5.1163, "yaw": 3.138952}

def timed(label, func, *args, **kwargs):
    started = time.perf_counter()
    try:
        return func(*args, **kwargs)
    finally:
        print(f"[TIMING] {label}: {time.perf_counter() - started:.2f}s")

def main():
    print("--- SYSTEM STARTUP ---")
    
    # Initialize speech and language components.
    ear = timed("Load wake-word detector", Ear)
    mouth = timed("Load speech output", Mouth)
    scribe = timed("Load transcription model", Scribe)
    recorder = timed("Load recorder and VAD", Recorder)
    # This ROS node provides the Nav2 action client used by the receptionist flow.
    receptionist_monitor = ReceptionistMonitor()
    receptionist_monitor.start()

    timed("Speak startup message", mouth.speak, "System online.")
    print("--- READY ---")

    while True:
        if timed("Wait for wake word", ear.wait_for_wake_word):
            # For this receptionist prototype, the wake word starts the flow.
            run_receptionist1(recorder, scribe, mouth, receptionist_monitor)
            print("--- WAITING ---")


def run_receptionist1(recorder, scribe, mouth, receptionist_monitor=None):
    """Welcome two guests, escort them through the task, and introduce them."""
    if receptionist_monitor is None:
        mouth.speak("I cannot start the receptionist task because navigation is unavailable.")
        return

    if not receptionist_monitor.wait_for_identity_ready(timeout=60.0):
        mouth.speak("Guest identity vision is not ready. Please check the camera and OSNet model.")
        return

    if not receptionist_monitor.wait_for_navigation(timeout=30.0):
        mouth.speak("Navigation is not ready. Please check Nav2 and try again.")
        return
    mouth.speak("I am going to the door.")
    reached_door = timed(
        "Navigate to door", receptionist_monitor.navigate_to_pose, **DOOR_POSE
    )
    if not reached_door:
        mouth.speak("I could not reach the door. Please check navigation.")
        return

    guests = []
    host_registration = {
        "enrolled": False,
        "guest_id": f"host_{int(time.time())}",
    }
    for guest_number in (1, 2):
        if guest_number == 2:
            mouth.speak("I will return to welcome the next guest.")
            reached_door = timed(
                "Return to door for second guest",
                receptionist_monitor.navigate_to_pose,
                **DOOR_POSE,
            )
            if not reached_door:
                mouth.speak("I could not return to the entrance for the next guest.")
                return

        guest = welcome_and_escort_guest(
            guest_number,
            recorder,
            scribe,
            mouth,
            receptionist_monitor,
            interest_at_door=(guest_number == 2),
            host_registration=host_registration,
        )
        if guest is None:
            return
        guests.append(guest)

    first, second = guests
    visible_guests = receptionist_monitor.wait_for_guest_identities(
        (first["guest_id"], second["guest_id"]), timeout=20.0
    )
    if visible_guests is None:
        mouth.speak(
            "I cannot confidently identify both guests, so I will not make introductions yet."
        )
        return
    for guest_id, people in visible_guests.items():
        print(f"[RECEPTIONIST1] Confirmed {guest_id} with identity state {people[0]['identity_state']}")
    mouth.speak(
        f"{first['name']}, this is {second['name']}. "
        f"{second['name']} likes {second['drink']} and is interested in {second['interest']}."
    )
    mouth.speak(
        f"{second['name']}, this is {first['name']}. "
        f"{first['name']} likes {first['drink']} and is interested in {first['interest']}."
    )


def welcome_and_escort_guest(
    guest_number,
    recorder,
    scribe,
    mouth,
    receptionist_monitor,
    interest_at_door=False,
    host_registration=None,
):
    """Collect one guest's details, show the drinks, and escort them to the seating area."""
    task = Receptionist1()
    recorder.calibrate()
    mouth.speak(task.start())

    while not task.complete:
        filename = timed("Record receptionist response", recorder.record, "command.wav")
        if not filename:
            recorder.calibrate()
            mouth.speak("I didn't hear you. What is your name?")
            continue

        try:
            guest_text = timed("Transcribe receptionist response", scribe.listen, filename)
            print(f"[RECEPTIONIST1 GUEST {guest_number}]: {guest_text}")
            reply = task.respond(guest_text)
            if reply:
                reply = normalize_spelling_for_speech(reply)
                if not task.complete:
                    recorder.calibrate()
                timed("Speak receptionist response", speak_receptionist_reply, mouth, reply)
        except Exception as exc:
            print(f"[RECEPTIONIST1 ERROR] {exc}")
            recorder.calibrate()
            mouth.speak("Sorry, could you please repeat that?")
        finally:
            if os.path.exists(filename):
                os.remove(filename)

    guest_id = next_guest_id()
    mouth.speak(
        "Please stand where I can see you clearly and turn slowly. I will register "
        "your appearance. If several new people are visible, I will register the "
        "person closest to the camera. Please stand closer than anyone else."
    )
    if not receptionist_monitor.enroll_guest_identity(guest_id, timeout=35.0):
        mouth.speak(
            "I could not identify you reliably, so I cannot safely continue the introductions."
        )
        return None
    try:
        save_guest_record(guest_id=guest_id, name=task.confirmed_name)
    except OSError as exc:
        print(f"[RECEPTIONIST1] Could not save guest identity record: {exc}")

    interest = None
    if interest_at_door:
        interest = ask_and_confirm_preference(
            "What is your interest?", "interest", recorder, scribe, mouth
        )
        if interest is None:
            return None

    mouth.speak("Please follow me to the drinks table.")
    if not timed("Navigate to drinks table", receptionist_monitor.navigate_to_pose, **TABLE_POSE):
        mouth.speak("I could not reach the drinks table. Please wait while I get help.")
        return None

    if not timed(
        "Turn to guest at drinks table",
        receptionist_monitor.navigate_to_pose,
        **TABLE_BACKWARD_POSE,
    ):
        mouth.speak("I could not turn toward you at the drinks table.")
        return None

    mouth.speak("Please answer after I complete my sentence.")
    if interest is None:
        interest = ask_and_confirm_preference(
            "What is your interest?", "interest", recorder, scribe, mouth
        )
        if interest is None:
            return None
    drink = ask_and_confirm_preference(
        "What is your favorite drink?", "favorite drink", recorder, scribe, mouth
    )
    if drink is None:
        return None

    if not timed("Face drinks table", receptionist_monitor.navigate_to_pose, **TABLE_POSE):
        mouth.speak("I could not turn back to the drinks table.")
        return None
    try:
        found, observed, favorite_position = timed(
            "Search for favorite drink", find_favorite_drink, drink
        )
    except Exception as exc:
        print(f"[RECEPTIONIST1] Drink search failed: {exc}")
        mouth.speak("I could not start the drink search. Please wait while I get help.")
        return None

    if found:
        drink_result_message = (
            f"Your favorite drink, {drink}, is on the {favorite_position} side of the table."
        )
    elif observed:
        position_phrases = {
            "left": "on the left", "center": "in the center", "right": "on the right"
        }
        labels = ", ".join(
            f"{label} {position_phrases[position]}"
            for label, position in sorted(observed.items())
        )
        drink_result_message = f"I could not find {drink}. I can see {labels} of the table."
    else:
        drink_result_message = f"I could not find {drink} on the table."

    if not timed(
        "Turn back to guest", receptionist_monitor.navigate_to_pose, **TABLE_BACKWARD_POSE
    ):
        mouth.speak("I could not turn back toward you.")
    mouth.speak(drink_result_message)
    mouth.speak(
        "Please follow me to the living area and stay behind me while I’m moving. "
        "Please don’t stand in front of me until I ask you to."
    )
    if not timed("Navigate to seating area", receptionist_monitor.navigate_to_pose, **SEATING_AREA_POSE):
        mouth.speak("I could not reach the seating area. Please wait while I get help.")
        return None

    if host_registration is not None and not host_registration["enrolled"]:
        host_id = host_registration["guest_id"]
        if receptionist_monitor.face_and_enroll_host(host_id, timeout=35.0):
            host_registration["enrolled"] = True
            print(f"[RECEPTIONIST1] Faced and enrolled host identity as {host_id}")
        else:
            print("[RECEPTIONIST1] Host face/enrollment failed; continuing guest seating flow.")

    available_seat = receptionist_monitor.get_available_seat(timeout=8.0)
    if available_seat is None:
        mouth.speak("I cannot see a free seat right now. Please show me an available place to sit.")
        return None

    center_x = int(available_seat.get("center_pixel_x", 320))
    image_width = max(1, int(available_seat.get("image_width", 640)))
    relative_center_x = center_x / image_width
    if relative_center_x < 1.0 / 3.0:
        seat_location = "on my left"
    elif relative_center_x > 2.0 / 3.0:
        seat_location = "on my right"
    else:
        seat_location = "in front of me"
    seat_id = available_seat.get("id")
    print(
        f"[RECEPTIONIST1] Guest {guest_number}: offering detected free "
        f"{available_seat.get('type', 'seat')} {seat_id} ({seat_location})"
    )
    mouth.speak(f"Please take the free seat {seat_location}.")
    if not receptionist_monitor.wait_for_guest_sitting(
        guest_id, available_seat, timeout=20.0
    ):
        mouth.speak("I could not confirm you are seated. Please sit in the indicated place.")
        return None

    guest = {
        "guest_id": guest_id,
        "name": task.confirmed_name,
        "interest": interest,
        "drink": drink,
    }
    try:
        save_guest_record(**guest)
        print(f"[RECEPTIONIST1] Saved {guest_id}: {guest['name']}")
    except OSError as exc:
        print(f"[RECEPTIONIST1] Could not save guest record: {exc}")
    return guest


def ask_and_confirm_preference(question, label, recorder, scribe, mouth):
    """Ask a guest preference, repeat it for confirmation, and return it."""
    while True:
        recorder.calibrate()
        mouth.speak(question)
        filename = timed(f"Record {label}", recorder.record, "command.wav")
        if not filename:
            mouth.speak("I didn't hear you. " + question)
            continue

        try:
            answer = timed(f"Transcribe {label}", scribe.listen, filename)
            answer = answer.strip().strip(" \t\r\n.,!?;:")
            answer = normalize_preference_answer(answer)
        except Exception as exc:
            print(f"[RECEPTIONIST1] Could not transcribe {label}: {exc}")
            mouth.speak("Sorry, could you repeat that?")
            continue
        finally:
            if os.path.exists(filename):
                os.remove(filename)

        if not answer:
            mouth.speak("I didn't catch that. " + question)
            continue

        while True:
            recorder.calibrate()
            spoken_label = "favorite drink" if label == "drink" else label
            mouth.speak(f"I heard your {spoken_label} is {answer}. Is that right?")
            filename = timed(f"Record {label} confirmation", recorder.record, "command.wav")
            if not filename:
                mouth.speak("Please answer yes or no.")
                continue

            try:
                confirmation = timed(
                    f"Transcribe {label} confirmation", scribe.listen, filename
                )
            except Exception as exc:
                print(f"[RECEPTIONIST1] Could not transcribe confirmation: {exc}")
                mouth.speak("Sorry, please answer yes or no.")
                continue
            finally:
                if os.path.exists(filename):
                    os.remove(filename)

            decision = Receptionist1._yes_no(confirmation)
            if decision == "yes":
                return answer
            if decision == "no":
                break
            mouth.speak("Please answer yes or no.")


def normalize_preference_answer(answer):
    """Remove common first-person framing while keeping the actual preference."""
    framing = re.compile(
        r"^(?:my interest is|my favorite drink is|i am interested in|"
        r"i'm interested in|i like to|i like|i enjoy|i love)\s+",
        re.IGNORECASE,
    )
    return framing.sub("", answer, count=1).strip(" \t\r\n.,!?;:")


def next_guest_id():
    """Allocate a stable logical identity without reusing a stored guest ID."""
    try:
        records = json.loads(GUEST_RECORDS_FILE.read_text(encoding="utf-8"))
        numbers = [
            int(item["guest_id"].split("_")[-1])
            for item in records
            if isinstance(item, dict)
            and str(item.get("guest_id", "")).startswith("guest_")
            and str(item["guest_id"]).split("_")[-1].isdigit()
        ]
    except (OSError, json.JSONDecodeError, TypeError):
        numbers = []
    return f"guest_{max(numbers, default=0) + 1}"


def save_guest_record(guest_id, name, interest=None, drink=None):
    """Persist confirmed guest details under the next guest_N identifier."""
    try:
        records = json.loads(GUEST_RECORDS_FILE.read_text(encoding="utf-8"))
        if not isinstance(records, list):
            records = []
    except (OSError, json.JSONDecodeError):
        records = []

    existing = next(
        (item for item in records if isinstance(item, dict) and item.get("guest_id") == guest_id),
        None,
    )
    if existing is None:
        existing = {
            "guest_id": guest_id,
            "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        records.append(existing)
    existing.update({
        "name": name,
        "interest": interest if interest is not None else existing.get("interest"),
        "favorite_drink": drink if drink is not None else existing.get("favorite_drink"),
    })
    GUEST_RECORDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    GUEST_RECORDS_FILE.write_text(
        json.dumps(records, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return guest_id


def is_receptionist1_trigger(user_text):
    """Recognize explicit task requests and common Whisper variations."""
    normalized = " ".join(user_text.lower().split())
    trigger_phrases = (
        "start receptionist1",
        "start receptionist one",
        "start receptionist task",
        "start reception task",
        "start task 1",
        "start task one",
        "start the task",
        "start task",
        "start the toss",  # Common transcription of "start the task".
    )
    return any(phrase in normalized for phrase in trigger_phrases)


def is_vision_question(user_text):
    """Recognize spoken requests for a concise description of the camera view."""
    normalized = " ".join(user_text.lower().strip(" .,!?").split())
    phrases = (
        "what do you see",
        "what can you see",
        "describe what you see",
        "describe the scene",
        "tell me what you see",
        "what is in front of you",
        "look around and describe",
    )
    return any(phrase in normalized for phrase in phrases)


def speak_receptionist_reply(mouth, reply):
    """Speak the complete reply in one Piper request to avoid model reloads."""
    spelling_pattern = re.compile(
        r"(?<![A-Za-z])(?:[A-Za-z](?:-[A-Za-z]){1,})"
        r"(?:\s+[A-Za-z](?:-[A-Za-z]){1,})*(?![A-Za-z])"
    )
    segments = []
    cursor = 0

    for match in spelling_pattern.finditer(reply):
        normal_speed_text = reply[cursor:match.start()]
        if normal_speed_text.strip():
            segments.append((normal_speed_text.strip(), 1.0))

        # Keep trailing punctuation with the spelling for natural intonation.
        end = match.end()
        while end < len(reply) and reply[end] in ".,?!:;":
            end += 1
        letters = match.group().replace("-", ", ") + reply[match.end():end]
        segments.append((letters.strip(), 1.6))
        cursor = end

    remaining = reply[cursor:]
    if remaining.strip():
        segments.append((remaining.strip(), 1.0))

    if segments:
        mouth.speak(" ".join(text for text, _ in segments))


def normalize_spelling_for_speech(reply):
    """Spell the model's proposed name explicitly in the confirmation question."""
    match = re.search(r"(?i)\bspelled\s+([^?.!]+)", reply)
    if not match:
        return reply

    target = match.group(1).strip()
    # The model may already have emitted letter-by-letter spelling.
    if re.search(r"\b[A-Za-z](?:-[A-Za-z]){1,}\b", target):
        return reply

    words = re.findall(r"[A-Za-z]+", target)
    if not words:
        return reply

    if all(len(word) == 1 for word in words):
        spelled = "-".join(word.upper() for word in words)
    else:
        spelled = " ".join("-".join(letter.upper() for letter in word) for word in words)

    start, end = match.span(1)
    raw_target = reply[start:end]
    leading_spaces = len(raw_target) - len(raw_target.lstrip())
    trailing_spaces = len(raw_target) - len(raw_target.rstrip())
    return (
        reply[:start]
        + raw_target[:leading_spaces]
        + spelled
        + (raw_target[-trailing_spaces:] if trailing_spaces else "")
        + reply[end:]
    )

if __name__ == "__main__":
    main()
