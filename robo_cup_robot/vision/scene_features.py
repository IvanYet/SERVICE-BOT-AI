"""Qwen3.5 vision model configuration and feature extraction."""

import json
import os
import time

import ollama

DEFAULT_VLM_MODEL = "qwen3.5:2b"
DEFAULT_PARAPHRASE_MODEL = os.getenv("RECEPTIONIST_LLM_MODEL", "qwen2.5:1.5b")
VLM_TIMEOUT_SECONDS = float(os.getenv("RECEPTIONIST_VLM_TIMEOUT", "120"))
VLM_NUM_THREADS = max(1, min(8, os.cpu_count() or 4))
_OLLAMA = ollama.Client(timeout=VLM_TIMEOUT_SECONDS)


def describe_person_features(image_bytes, model):
    """Return the four requested visible attributes in a predictable format."""
    prompt = (
        "Look at the most prominent visible person. Return only a JSON object with "
        "exactly these four keys: shirt_color, glasses, hair_color, clothing_pattern. "
        "Use a short color for shirt_color and hair_color, or unclear only if it is "
        "not visible. glasses must be yes or no. clothing_pattern must be exactly "
        "one of checkered, plain, striped, graphic T, or unclear. Select the closest "
        "matching category. Inspect the person in the image carefully; do not describe "
        "the room or add any other fields."
    )
    started = time.monotonic()
    response = _OLLAMA.chat(
        model=model,
        think=False,
        format="json",
        messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
        options={
            "num_gpu": 0,
            "num_thread": VLM_NUM_THREADS,
            "num_ctx": 1024,
            "num_predict": 128,
            "temperature": 0,
        },
        keep_alive="30s",
    )
    content = response.get("message", {}).get("content", "")
    content = content.strip()
    if not content:
        thinking = response.get("message", {}).get("thinking", "").strip()
        detail = "reasoning without an answer" if thinking else "an empty answer"
        elapsed = time.monotonic() - started
        raise RuntimeError(f"Qwen VLM returned {detail} after {elapsed:.1f}s")
    if content.startswith("```"):
        content = content.split("\n", 1)[-1]
        if content.endswith("```"):
            content = content[:-3].strip()
    try:
        features = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        features = {}

    shirt = str(features.get("shirt_color", "unclear")).strip() or "unclear"
    hair = str(features.get("hair_color", "unclear")).strip() or "unclear"
    glasses_value = features.get("glasses", "unclear")
    if isinstance(glasses_value, bool):
        glasses = "yes" if glasses_value else "no"
    else:
        glasses = str(glasses_value).strip().lower()
        if glasses in ("true", "yes"):
            glasses = "yes"
        elif glasses in ("false", "no"):
            glasses = "no"
        else:
            glasses = "unclear"

    pattern = str(features.get("clothing_pattern", "unclear")).strip().lower()
    if "check" in pattern or "plaid" in pattern:
        pattern = "checkered"
    elif "stripe" in pattern:
        pattern = "striped"
    elif "plain" in pattern:
        pattern = "plain"
    elif "graphic" in pattern or "print" in pattern or "logo" in pattern:
        pattern = "graphic T"
    else:
        pattern = "unclear"

    return (
        f"Shirt color: {shirt}\n"
        f"Glasses: {glasses}\n"
        f"Hair color: {hair}\n"
        f"Clothing pattern: {pattern}"
    )


def paraphrase_person_features(features, model=None):
    """Turn verified VLM attributes into one short paragraph without adding facts."""
    model_name = model or DEFAULT_PARAPHRASE_MODEL
    response = ollama.Client(
        timeout=float(os.getenv("RECEPTIONIST_LLM_TIMEOUT", "60"))
    ).chat(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite the supplied appearance attributes as one short, natural paragraph. "
                    "Use only the supplied facts. Do not infer age, identity, personality, or other "
                    "details. Preserve unclear values as unknown. Do not use a list or headings."
                ),
            },
            {
                "role": "user",
                "content": f"Appearance attributes:\n{features}\n\nWrite one paragraph.",
            },
        ],
        options={
            "temperature": 0,
            "num_predict": 96,
            "num_ctx": 512,
            "num_thread": VLM_NUM_THREADS,
        },
        keep_alive="30s",
    )
    paragraph = response.get("message", {}).get("content", "").strip()
    paragraph = " ".join(paragraph.split())
    if not paragraph:
        raise RuntimeError("The language model returned an empty appearance paragraph")
    return paragraph
