# score.py

from typing import Tuple
import numpy as np


def score(text: str, model, threshold: float) -> Tuple[bool, float]:
    """
    Scores a trained sklearn model on input text.

    Args:
        text (str): input text
        model: trained sklearn estimator with predict_proba
        threshold (float): classification threshold

    Returns:
        prediction (bool): True (1) if spam else False (0)
        propensity (float): probability of positive class
    """

    if not isinstance(text, str):
        raise ValueError("text must be a string")

    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")

    # model expects iterable input
    prob = model.predict_proba([text])[0][1]

    prediction = prob >= threshold

    return bool(prediction), float(prob)