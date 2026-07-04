"""
Pure helper functions: input validation, formatting, and result shaping.

Keeping these free of Streamlit and model dependencies makes them trivial
to unit test in isolation (see tests/test_utils.py).
"""

from __future__ import annotations

from typing import Any

from core.config import APP_CONFIG, DEFAULT_STYLE, EMOTION_STYLE


class ValidationError(ValueError):
    """Raised when user-supplied text fails validation before inference."""


def validate_text(text: str) -> str:
    """
    Validate and normalize raw text input before it reaches the model.

    Raises:
        ValidationError: if the text is empty or exceeds the configured
            character limit.
    """
    if text is None:
        raise ValidationError("Please enter some text.")

    cleaned = text.strip()

    if len(cleaned) < APP_CONFIG.min_chars:
        raise ValidationError("Please enter some text.")

    if len(cleaned) > APP_CONFIG.max_chars:
        raise ValidationError(
            f"Text is too long ({len(cleaned)} characters). "
            f"Please limit input to {APP_CONFIG.max_chars} characters."
        )

    return cleaned


def get_emotion_style(label: str) -> dict[str, str]:
    """Look up the emoji/color styling for a given emotion label."""
    return EMOTION_STYLE.get(label.lower(), DEFAULT_STYLE)


def format_percentage(score: float) -> str:
    """Format a 0-1 confidence score as a human-readable percentage string."""
    return f"{round(score * 100, 2)}%"


def is_low_confidence(score: float) -> bool:
    """True if the top prediction's confidence is below the configured threshold."""
    return score < APP_CONFIG.low_confidence_threshold


def sort_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort raw pipeline predictions by score, descending."""
    return sorted(predictions, key=lambda item: item["score"], reverse=True)


def predictions_to_rows(text: str, predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Flatten a single text's predictions into row dicts suitable for a
    pandas DataFrame / CSV export, one row per (text, emotion) pair.
    """
    rows = []
    for pred in predictions:
        rows.append(
            {
                "text": text,
                "emotion": pred["label"],
                "confidence": round(pred["score"], 4),
            }
        )
    return rows
