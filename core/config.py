"""
Centralized configuration for the Emotion AI App.

Keeping constants in one place avoids "magic values" scattered across the
codebase and makes it trivial to tune behaviour (model choice, thresholds,
input limits) without touching engine or UI code.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Hugging Face emotion classification model."""

    model_name: str = "j-hartmann/emotion-english-distilroberta-base"
    task: str = "text-classification"
    return_all_scores: bool = True  # maps to top_k=None in the pipeline call
    device: int = -1  # -1 = CPU. Set to 0 for GPU if CUDA is available.


@dataclass(frozen=True)
class AppConfig:
    """Application-level limits and behavioural settings."""

    max_chars: int = 2000
    min_chars: int = 1
    max_batch_rows: int = 200
    low_confidence_threshold: float = 0.35  # below this, flag result as uncertain
    page_title: str = "Emotion AI"
    page_icon: str = "🎭"


# Emoji + hex color per emotion label, used for both the UI badges and charts.
# Colors chosen for reasonable colorblind-safe contrast (Okabe-Ito inspired).
EMOTION_STYLE: dict[str, dict[str, str]] = {
    "joy": {"emoji": "😄", "color": "#F2C14E"},
    "sadness": {"emoji": "😢", "color": "#5B84B1"},
    "anger": {"emoji": "😠", "color": "#D64550"},
    "fear": {"emoji": "😨", "color": "#8E7CC3"},
    "surprise": {"emoji": "😲", "color": "#59A96A"},
    "disgust": {"emoji": "🤢", "color": "#7B9E45"},
    "neutral": {"emoji": "😐", "color": "#9AA5B1"},
}

DEFAULT_STYLE = {"emoji": "🎭", "color": "#9AA5B1"}

MODEL_CONFIG = ModelConfig()
APP_CONFIG = AppConfig()
