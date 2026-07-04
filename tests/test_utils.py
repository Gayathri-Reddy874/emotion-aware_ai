import pytest

from core.config import APP_CONFIG
from core.utils import (
    ValidationError,
    format_percentage,
    get_emotion_style,
    is_low_confidence,
    predictions_to_rows,
    sort_predictions,
    validate_text,
)


class TestValidateText:
    def test_valid_text_is_stripped(self):
        assert validate_text("  hello world  ") == "hello world"

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError):
            validate_text("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError):
            validate_text("     ")

    def test_none_raises(self):
        with pytest.raises(ValidationError):
            validate_text(None)

    def test_too_long_raises(self):
        with pytest.raises(ValidationError):
            validate_text("a" * (APP_CONFIG.max_chars + 1))

    def test_max_length_boundary_is_allowed(self):
        text = "a" * APP_CONFIG.max_chars
        assert validate_text(text) == text


class TestEmotionStyle:
    def test_known_label_returns_specific_style(self):
        style = get_emotion_style("joy")
        assert style["emoji"] == "😄"

    def test_unknown_label_returns_default(self):
        style = get_emotion_style("bewilderment")
        assert style["emoji"] == "🎭"

    def test_case_insensitive(self):
        assert get_emotion_style("JOY") == get_emotion_style("joy")


class TestFormatting:
    def test_format_percentage(self):
        assert format_percentage(0.4567) == "45.67%"

    def test_format_percentage_zero(self):
        assert format_percentage(0.0) == "0.0%"

    def test_low_confidence_below_threshold(self):
        assert is_low_confidence(APP_CONFIG.low_confidence_threshold - 0.01) is True

    def test_low_confidence_above_threshold(self):
        assert is_low_confidence(APP_CONFIG.low_confidence_threshold + 0.01) is False


class TestSortPredictions:
    def test_sorts_descending_by_score(self):
        raw = [
            {"label": "sadness", "score": 0.1},
            {"label": "joy", "score": 0.9},
            {"label": "anger", "score": 0.5},
        ]
        sorted_preds = sort_predictions(raw)
        assert [p["label"] for p in sorted_preds] == ["joy", "anger", "sadness"]


class TestPredictionsToRows:
    def test_flattens_correctly(self):
        preds = [{"label": "joy", "score": 0.87654}]
        rows = predictions_to_rows("I'm happy", preds)
        assert rows == [{"text": "I'm happy", "emotion": "joy", "confidence": 0.8765}]
