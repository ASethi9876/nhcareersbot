
import os
import requests

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

API_URL = f"https://api-inference.huggingface.co/models/{MODEL_NAME}"
HEADERS = {
    "Authorization": f"Bearer {HF_API_TOKEN}",
    "Content-Type": "application/json",
}

def generate(prompt: str) -> str:
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 120,
            "temperature": 0.3,
            "top_p": 0.9,
            "return_full_text": False
        }
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=30
    )
    response.raise_for_status()

    data = response.json()

    # HF Inference API returns a list for text-generation models
    if isinstance(data, list) and "generated_text" in data[0]:
        return data[0]["generated_text"].strip()

    return "Sorry, I’m having trouble answering right now."
