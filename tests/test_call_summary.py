import builtins

import call_summary


def test_text_emotion_uses_lightweight_fallback_without_transformers(
    monkeypatch,
) -> None:
    monkeypatch.delenv("CALL_SUMMARY_ENABLE_TRANSFORMERS", raising=False)
    call_summary._models.clear()

    real_import = builtins.__import__

    def fail_on_transformers(name, *args, **kwargs):
        if name == "transformers":
            raise AssertionError("transformers should not load by default")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_on_transformers)

    result = call_summary._run_text_emotion("الشقة ممتازة ومناسبة جدا", "ar")

    assert result["dominant_emotion"] == "positive"
    assert result["sentiment"] == "positive"


def test_text_models_are_opt_in(monkeypatch) -> None:
    monkeypatch.delenv("CALL_SUMMARY_ENABLE_TRANSFORMERS", raising=False)
    call_summary._models.clear()

    assert call_summary._load_text_models() is False
    assert call_summary._models["text_models_unavailable"] is True
