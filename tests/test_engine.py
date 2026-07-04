import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from core.engine import EmotionEngine, InferenceError, ModelLoadError


def _fake_transformers_module(pipeline_factory) -> types.ModuleType:
    """
    Build a stand-in `transformers` module exposing a `pipeline` callable,
    so tests don't require the real (multi-GB, torch-dependent) package.
    """
    fake_module = types.ModuleType("transformers")
    fake_module.pipeline = pipeline_factory
    return fake_module


def make_loaded_engine(pipeline_return_value):
    """Helper: build an EmotionEngine with a mocked transformers pipeline."""
    engine = EmotionEngine()
    mock_pipeline = MagicMock(return_value=pipeline_return_value)
    fake_module = _fake_transformers_module(MagicMock(return_value=mock_pipeline))
    with patch.dict(sys.modules, {"transformers": fake_module}):
        engine.load()
    return engine, mock_pipeline


class TestLoad:
    def test_load_success_sets_is_loaded(self):
        engine, _ = make_loaded_engine([[{"label": "joy", "score": 0.9}]])
        assert engine.is_loaded is True

    def test_load_failure_raises_model_load_error(self):
        engine = EmotionEngine()
        fake_module = _fake_transformers_module(MagicMock(side_effect=OSError("network error")))
        with patch.dict(sys.modules, {"transformers": fake_module}):
            with pytest.raises(ModelLoadError):
                engine.load()

    def test_load_is_idempotent(self):
        engine, _ = make_loaded_engine([[{"label": "joy", "score": 0.9}]])
        first_pipeline = engine._pipeline
        engine.load()  # second call should be a no-op
        assert engine._pipeline is first_pipeline


class TestPredict:
    def test_predict_before_load_raises(self):
        engine = EmotionEngine()
        with pytest.raises(InferenceError):
            engine.predict("hello")

    def test_predict_returns_sorted_results(self):
        raw_output = [[
            {"label": "sadness", "score": 0.1},
            {"label": "joy", "score": 0.85},
            {"label": "neutral", "score": 0.05},
        ]]
        engine, _ = make_loaded_engine(raw_output)
        results = engine.predict("I'm thrilled today")
        assert results[0]["label"] == "joy"
        assert results[0]["score"] == 0.85

    def test_predict_handles_flat_list_output(self):
        # Some pipeline configs return a flat list instead of nested.
        raw_output = [{"label": "anger", "score": 0.7}]
        engine, mock_pipeline = make_loaded_engine(raw_output)
        mock_pipeline.return_value = raw_output
        results = engine.predict("This is infuriating")
        assert results[0]["label"] == "anger"

    def test_predict_wraps_model_exception(self):
        engine = EmotionEngine()
        mock_pipeline = MagicMock(side_effect=RuntimeError("CUDA out of memory"))
        fake_module = _fake_transformers_module(MagicMock(return_value=mock_pipeline))
        with patch.dict(sys.modules, {"transformers": fake_module}):
            engine.load()
        with pytest.raises(InferenceError):
            engine.predict("some text")


class TestPredictBatch:
    def test_batch_preserves_order(self):
        raw_output = [[{"label": "joy", "score": 0.9}]]
        engine, _ = make_loaded_engine(raw_output)
        results = engine.predict_batch(["text one", "text two", "text three"])
        assert len(results) == 3
        assert all(r[0]["label"] == "joy" for r in results)

    def test_batch_continues_after_single_item_failure(self):
        engine = EmotionEngine()
        mock_pipeline = MagicMock(side_effect=[RuntimeError("boom"), [[{"label": "joy", "score": 0.9}]]])
        fake_module = _fake_transformers_module(MagicMock(return_value=mock_pipeline))
        with patch.dict(sys.modules, {"transformers": fake_module}):
            engine.load()
        results = engine.predict_batch(["bad text", "good text"])
        assert results[0] == []  # failed item returns empty list
        assert results[1][0]["label"] == "joy"
