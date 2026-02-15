import requests
import os
import base64
import json

# Base URL of your running FastAPI server
BASE_SERVER_URL = "https://8000-dep-01khgcb8hf1kcdc87pbkv4bfz1-d.cloudspaces.litng.ai/"

IMAGE_PATH = "test.jpg"
TEXT_PROMPT = "watches"

BOUNDING_BOXES = [
    [4207.0, 3034.0, 238.0, 492.0],
    [2951.0, 2320.0, 219.0, 403.0]
]

BOUNDING_BOX_LABELS = [True, False]


# -------------------------
# Utilities
# -------------------------

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def make_prediction_request(endpoint, payload):
    try:
        response = requests.post(
            f"{BASE_SERVER_URL}{endpoint}",
            json=payload  # use json= instead of manual dumps
        )
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        print(f"❌ Could not connect to {BASE_SERVER_URL}. Is the server running?")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP error: {e}")
        print("Response text:", response.text)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    return None


def process_results(result):
    if not isinstance(result, dict):
        print(f"❌ Unexpected response format: {type(result)}")
        print(result)
        return None

    print("✅ Prediction successful!")
    print(f"Number of masks: {len(result.get('masks', []))}")
    print(f"Number of boxes: {len(result.get('boxes', []))}")
    print(f"Number of scores: {len(result.get('scores', []))}")

    masks = result.get("masks", [])
    boxes = result.get("boxes", [])
    scores = result.get("scores", [])

    return masks, boxes, scores

# -------------------------
# Text Prompt Segmentation
# -------------------------

def segment_image_with_text_prompt(image_path, text_prompt):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    print(f"\nSending image + text prompt to /predict-image-text")

    image_b64 = encode_image_to_base64(image_path)

    payload = {
        "image": image_b64,
        "prompt": text_prompt
    }

    result = make_prediction_request("/predict-image-text", payload)

    if result is not None:
        return process_results(result)

# -------------------------
# Bounding Box Segmentation
# -------------------------

def segment_image_with_bounding_box(image_path, boxes, labels):
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return

    print(f"\nSending image + bounding boxes to /predict-bounding-box")

    image_b64 = encode_image_to_base64(image_path)

    payload = {
        "image": image_b64,
        "boxes": boxes,
        "labels": labels
    }

    result = make_prediction_request("/predict-bounding-box", payload)

    if result is not None:
        return process_results(result)


# -------------------------
# Run Examples
# -------------------------

if __name__ == "__main__":

    print("\n--- Running Text-Based Segmentation ---")
    segment_image_with_text_prompt(IMAGE_PATH, TEXT_PROMPT)

    # Uncomment if needed
    # print("\n--- Running Bounding Box Segmentation ---")
    # segment_image_with_bounding_box(IMAGE_PATH, BOUNDING_BOXES, BOUNDING_BOX_LABELS)
