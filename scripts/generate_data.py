import os
import json
import time
import random
from collections import Counter
from tqdm import tqdm
from google import genai
from google.genai import types
from dotenv import load_dotenv


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

NUM_LABELS = 250
LABEL_BATCH_SIZE = 50

NUM_SAMPLES = 4000
BATCH_SIZE = 20

OUTPUT_PATH = "data/synthetic_data.json"
LABEL_PATH = "data/label_pool.json"

MODEL_ID = "gemini-2.5-flash-lite"

MAX_RETRIES = 3
REQUEST_DELAY = 8  



if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=API_KEY)

os.makedirs("data", exist_ok=True)



def safe_parse_json(text):
    try:
        start = text.find("[")
        end = text.rfind("]") + 1
        if start == -1 or end == -1:
            return None
        return json.loads(text[start:end])
    except Exception:
        return None



def generate_label_batch(batch_size):
    prompt = f"""
Generate {batch_size} diverse topic labels for open-vocabulary text classification.

Requirements:
- Mix styles:
  • Technical terms
  • Industry sectors
  • Scientific concepts
  • Policy areas
  • Social themes

- Avoid repeating patterns like "AI for X".
- Ensure at least 40% do NOT contain the word "AI".
- Vary abstraction levels.
- Return ONLY a valid JSON list of strings.
"""

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.8,
                ),
            )

            parsed = safe_parse_json(response.text)
            if parsed:
                return parsed

        except Exception as e:
            print(f"Label batch retry {attempt+1}/{MAX_RETRIES}: {e}")
            time.sleep(3)

    return None


def get_label_pool():
    if os.path.exists(LABEL_PATH):
        with open(LABEL_PATH, "r") as f:
            print("Loaded existing label pool.")
            return json.load(f)

    print("Generating label pool in batches...")

    labels = []

    while len(labels) < NUM_LABELS:
        batch = generate_label_batch(LABEL_BATCH_SIZE)

        if batch:
            labels.extend(batch)
            print(f"Collected {len(labels)} labels...")

        time.sleep(REQUEST_DELAY)


    labels = list(set([
        l.strip()
        for l in labels
        if isinstance(l, str) and len(l.split()) >= 2
    ]))[:NUM_LABELS]

    with open(LABEL_PATH, "w") as f:
        json.dump(labels, f, indent=2)

    print(f"Final label pool size: {len(labels)}")
    return labels


DATA_PROMPT_TEMPLATE = """
Generate {n} unique training examples for open-vocabulary zero-shot classification.

Use ONLY these labels:
{label_subset}

Rules:
- "text": A realistic sentence (15-30 words).
- Use diverse tones (news, technical, academic, casual).
- "labels": 2-3 relevant labels from provided subset.
- Avoid trivial matches.
- Return ONLY valid JSON list.
"""


def generate_data_batch(label_subset, n):
    prompt = DATA_PROMPT_TEMPLATE.format(
        label_subset=", ".join(label_subset),
        n=n
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.9,
                ),
            )

            parsed = safe_parse_json(response.text)
            if parsed:
                return parsed

        except Exception as e:
            print(f"Data batch retry {attempt+1}/{MAX_RETRIES}: {e}")
            time.sleep(3)

    return None


def main():
    label_pool = get_label_pool()

    all_samples = []
    seen_texts = set()

    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r") as f:
            all_samples = json.load(f)
            seen_texts = {s["text"] for s in all_samples}
        print(f"Resuming from {len(all_samples)} samples...")

    remaining = NUM_SAMPLES - len(all_samples)

    if remaining <= 0:
        print("Dataset already complete.")
        return

    pbar = tqdm(total=NUM_SAMPLES)
    pbar.update(len(all_samples))

    while len(all_samples) < NUM_SAMPLES:

        subset = random.sample(label_pool, min(50, len(label_pool)))

        batch_data = generate_data_batch(subset, BATCH_SIZE)

        if batch_data is None:
            print("Stopping due to repeated failures.")
            break

        for item in batch_data:
            text = item.get("text", "").strip()
            labels = item.get("labels", [])

            if (
                text
                and text not in seen_texts
                and isinstance(labels, list)
            ):
                valid_labels = [l for l in labels if l in label_pool]

                if len(valid_labels) >= 1:
                    all_samples.append({
                        "text": text,
                        "labels": valid_labels
                    })
                    seen_texts.add(text)
                    pbar.update(1)

        with open(OUTPUT_PATH, "w") as f:
            json.dump(all_samples, f, indent=2)

        time.sleep(REQUEST_DELAY)

    pbar.close()

    print(f"\nFinal Dataset Size: {len(all_samples)}")

    label_counts = Counter([
        l for s in all_samples for l in s["labels"]
    ])
    print("\nTop 5 Labels:", label_counts.most_common(5))


if __name__ == "__main__":
    main()