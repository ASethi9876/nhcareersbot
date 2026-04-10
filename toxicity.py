from detoxify import Detoxify

toxicity_model = Detoxify("original")

def is_inappropriate(text: str, threshold: float = 0.6) -> bool:
    scores = toxicity_model.predict(text)
    return max(scores.values()) > threshold
