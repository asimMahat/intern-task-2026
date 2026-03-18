"""Integration tests -- require OPENAI_API_KEY to be set.

Run with: pytest tests/test_feedback_integration.py -v

These tests make real API calls. Skip them in CI or when no key is available.
"""

import os

import pytest
from app.feedback import get_feedback
from app.models import FeedbackRequest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set -- skipping integration tests",
)

VALID_ERROR_TYPES = {
    "grammar",
    "spelling",
    "word_choice",
    "punctuation",
    "word_order",
    "missing_word",
    "extra_word",
    "conjugation",
    "gender_agreement",
    "number_agreement",
    "tone_register",
    "other",
}
VALID_DIFFICULTIES = {"A1", "A2", "B1", "B2", "C1", "C2"}


def _assert_valid_response(result, expect_correct: bool):
    """Shared assertions for every integration test."""
    assert result.is_correct is expect_correct
    assert result.difficulty in VALID_DIFFICULTIES
    for error in result.errors:
        assert error.error_type in VALID_ERROR_TYPES
        assert len(error.original) > 0
        assert len(error.correction) > 0
        assert len(error.explanation) > 0
    if expect_correct:
        assert result.errors == []
    else:
        assert len(result.errors) >= 1


# ---------------------------------------------------------------------------
# 1. Spanish conjugation error (single error)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)


# ---------------------------------------------------------------------------
# 2. Correct German sentence (edge case: no errors)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_correct_german_sentence():
    request = FeedbackRequest(
        sentence="Ich habe gestern einen interessanten Film gesehen.",
        target_language="German",
        native_language="English",
    )
    result = await get_feedback(request)
    _assert_valid_response(result, expect_correct=True)
    assert result.corrected_sentence == request.sentence


# ---------------------------------------------------------------------------
# 3. French gender agreement (multiple errors, same type)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_french_gender_agreement():
    result = await get_feedback(
        FeedbackRequest(
            sentence="La chat noir est sur le table.",
            target_language="French",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    assert len(result.errors) >= 2


# ---------------------------------------------------------------------------
# 4. Portuguese spelling + grammar (multiple errors, different types)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portuguese_mixed_error_types():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Eu quero comprar um prezente para minha irma, mas nao sei o que ela gosta.",
            target_language="Portuguese",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    error_types = {e.error_type for e in result.errors}
    assert len(error_types) >= 2, (
        f"Expected at least 2 distinct error types, got: {error_types}"
    )


# ---------------------------------------------------------------------------
# 5. Italian word order error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_italian_word_order():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Mangiato ho la pizza ieri.",
            target_language="Italian",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)


# ---------------------------------------------------------------------------
# 6. Spanish missing word (missing preposition)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spanish_missing_word():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Voy la tienda.",
            target_language="Spanish",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)


# ---------------------------------------------------------------------------
# 7. German extra word (duplicated word)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_german_extra_word():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Ich bin bin muede.",
            target_language="German",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)


# ---------------------------------------------------------------------------
# 8. Correct Spanish sentence (second correct-sentence edge case)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_correct_spanish_sentence():
    request = FeedbackRequest(
        sentence="Me gusta mucho la comida mexicana.",
        target_language="Spanish",
        native_language="English",
    )
    result = await get_feedback(request)
    _assert_valid_response(result, expect_correct=True)
    assert result.corrected_sentence == request.sentence


# ---------------------------------------------------------------------------
# 9. Explanation language regression (cross-lingual drift)
#    Ensures explanations are in English when native_language="English",
#    not in the target language. See BUG_REPORT.md for details.
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explanation_language_english():
    result = await get_feedback(
        FeedbackRequest(
            sentence="Yo soy fue al mercado ayer.",
            target_language="Spanish",
            native_language="English",
        )
    )
    _assert_valid_response(result, expect_correct=False)
    for error in result.errors:
        has_ascii = any(c.isascii() and c.isalpha() for c in error.explanation)
        assert has_ascii, (
            f"Explanation appears to not be in English: {error.explanation!r}"
        )
