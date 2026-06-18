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


def test_auto_outcome_qualifies_viewing_request_even_with_no_word() -> None:
    builder = call_summary.CallSummaryBuilder(phone="+201019945011")
    rows = [
        {"text": "لا انا عاوزه اشوفها الاول", "intent": "general_question"},
        {"text": "اقرب وقت", "intent": "schedule_visit"},
    ]

    assert builder._auto_determine_outcome(rows) == "qualified"


def test_auto_outcome_unqualifies_explicit_rejection() -> None:
    builder = call_summary.CallSummaryBuilder()
    rows = [{"text": "مش مهتم شكرا", "intent": "unqualified"}]

    assert builder._auto_determine_outcome(rows) == "unqualified"
