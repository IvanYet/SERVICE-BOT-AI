"""Test the drink dialogue, RealSense stream, and YOLO search without navigation."""

from robo_cup_robot.speech.mouth import Mouth
from robo_cup_robot.speech.recorder import Recorder
from robo_cup_robot.speech.whisper import Scribe
from robo_cup_robot.task_manager.main import ask_and_confirm_preference
from robo_cup_robot.vision.drink_finder import find_favorite_drink


def main():
    mouth = Mouth()
    recorder = Recorder()
    scribe = Scribe()

    mouth.speak("Starting the camera and drink detection test.")
    favorite = ask_and_confirm_preference(
        "What is your favorite drink?", "favorite drink", recorder, scribe, mouth
    )

    mouth.speak("I am checking the camera for your drink. Please wait.")
    found, observed, position = find_favorite_drink(favorite)

    if found:
        message = (
            f"Your favorite drink, {favorite}, is on the {position} side of the table."
        )
    elif observed:
        position_phrases = {
            "left": "on the left",
            "center": "in the center",
            "right": "on the right",
        }
        drinks = ", ".join(
            f"{label} {position_phrases[drink_position]}"
            for label, drink_position in sorted(observed.items())
        )
        message = f"I could not find {favorite}. I can see {drinks} of the table."
    else:
        message = f"I could not find {favorite} on the table."

    print(f"[DRINK TEST] Favorite: {favorite}")
    print(f"[DRINK TEST] Found: {found}; positions: {observed}")
    mouth.speak(message)


if __name__ == "__main__":
    main()
