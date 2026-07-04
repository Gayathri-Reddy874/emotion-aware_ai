"""
Model loading and inference logic for emotion classification.

This module isolates all Hugging Face / transformers interaction so the
UI layer (app.py) never touches the model directly. That separation means:
  - the model can be swapped or upgraded without touching app.py
  - engine logic is unit-testable with a mocked pipeline
  - inference errors are caught and surfaced as typed exceptions instead
    of raw stack traces reaching the user
"""

from __future__ import annotations

import logging
from typing import Any

from core.config import MODEL_CONFIG
from core.utils import sort_predictions

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when the underlying transformers pipeline fails to load."""


class InferenceError(RuntimeError):
    """Raised when the model fails to produce a prediction for valid input."""


class EmotionEngine:
    """
    Thin wrapper around a Hugging Face text-classification pipeline,
    specialized for multi-label emotion detection.
    """

    def __init__(self) -> None:
        self._pipeline = None

    def load(self) -> "EmotionEngine":
        """
        Lazily load the underlying transformers pipeline.

        Returns self so callers can chain: EmotionEngine().load()
        """
        if self._pipeline is not None:
            return self

        try:
            from transformers import pipeline  # local import: heavy dependency

            self._pipeline = pipeline(
                MODEL_CONFIG.task,
                model=MODEL_CONFIG.model_name,
                top_k=None if MODEL_CONFIG.return_all_scores else 1,
                device=MODEL_CONFIG.device,
            )
            logger.info("Loaded model '%s' successfully.", MODEL_CONFIG.model_name)
        except Exception as exc:  # noqa: BLE001 - we re-raise as a typed error
            logger.exception("Failed to load emotion classification model.")
            raise ModelLoadError(
                "Could not load the emotion classification model. "
                "This is usually caused by a network issue on first download, "
                "or missing disk space. See logs for details."
            ) from exc

        return self

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None

    def predict(self, text: str) -> list[dict[str, Any]]:
        """
        Run inference on a single piece of text.

        Returns a list of {"label": str, "score": float} dicts sorted by
        score descending. Raises InferenceError on failure.
        """
        if self._pipeline is None:
            raise InferenceError("Engine used before load() was called.")

        try:
            raw = self._pipeline(text)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed for input of length %d.", len(text))
            raise InferenceError(
                "The model failed to analyze this text. Try shortening it "
                "or removing unusual characters."
            ) from exc

        # top_k=None returns a nested list: [[{...}, {...}, ...]]
        predictions = raw[0] if isinstance(raw[0], list) else raw
        return sort_predictions(predictions)

    def predict_batch(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """
        Run inference on multiple texts, preserving input order.
        A failure on one item does not abort the whole batch; that item's
        result list is returned empty and the caller can flag it.
        """
        results: list[list[dict[str, Any]]] = []
        for text in texts:
            try:
                results.append(self.predict(text))
            except InferenceError:
                results.append([])
        return results
