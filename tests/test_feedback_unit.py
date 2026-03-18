"""Unit tests -- run without an API key using mocked LLM responses."""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app import feedback
from app.feedback import get_feedback
from app.models import FeedbackRequest


@pytest.fixture(autouse=True)
def _clear_feedback_cache():
    """Clear the in-memory LRU cache between tests."""
    feedback._cache.clear()


def _mock_completion(response_data: dict) -> MagicMock:
    """Build a mock ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = json.dumps(response_data)
    completion = MagicMock()
    completion.choices = [choice]
    return completion


async def _run_mocked_feedback(mock_response, sentence, target_language, native_language):
    """Patch AsyncOpenAI, inject mock_response, and return the parsed result."""
    with patch("app.feedback.AsyncOpenAI") as MockClient:
        instance = MockClient.return_value
        instance.chat.completions.create = AsyncMock(
            return_value=_mock_completion(mock_response)
        )
        request = FeedbackRequest(
            sentence=sentence,
            target_language=target_language,
            native_language=native_language,
        )
        return await get_feedback(request)


# ---------------------------------------------------------------------------
# 1. Spanish conjugation error (single error)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spanish_conjugation_error():
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": "You mixed two verb forms.",
            }
        ],
        "difficulty": "A2",
    }

    result = await _run_mocked_feedback(
        mock_response, "Yo soy fue al mercado ayer.", "Spanish", "English"
    )

    assert result.is_correct is False
    assert result.corrected_sentence == "Yo fui al mercado ayer."
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "conjugation"
    assert result.difficulty == "A2"


# ---------------------------------------------------------------------------
# 2. Correct German sentence (edge case: no errors)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_correct_german_sentence():
    sentence = "Ich habe gestern einen interessanten Film gesehen."
    mock_response = {
        "corrected_sentence": sentence,
        "is_correct": True,
        "errors": [],
        "difficulty": "B1",
    }

    result = await _run_mocked_feedback(mock_response, sentence, "German", "English")

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == sentence


# ---------------------------------------------------------------------------
# 3. French gender agreement (multiple errors, same type)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_french_multiple_gender_errors():
    mock_response = {
        "corrected_sentence": "Le chat noir est sur la table.",
        "is_correct": False,
        "errors": [
            {
                "original": "La chat",
                "correction": "Le chat",
                "error_type": "gender_agreement",
                "explanation": "'Chat' is masculine in French.",
            },
            {
                "original": "le table",
                "correction": "la table",
                "error_type": "gender_agreement",
                "explanation": "'Table' is feminine in French.",
            },
        ],
        "difficulty": "A1",
    }

    result = await _run_mocked_feedback(
        mock_response, "La chat noir est sur le table.", "French", "English"
    )

    assert result.is_correct is False
    assert len(result.errors) == 2
    assert all(e.error_type == "gender_agreement" for e in result.errors)


# ---------------------------------------------------------------------------
# 4. Portuguese spelling + grammar (multiple errors, different types)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_portuguese_mixed_error_types():
    mock_response = {
        "corrected_sentence": "Eu quero comprar um presente para minha irmã, mas não sei do que ela gosta.",
        "is_correct": False,
        "errors": [
            {
                "original": "prezente",
                "correction": "presente",
                "error_type": "spelling",
                "explanation": "'Present/gift' in Portuguese is spelled 'presente' with an 's', not a 'z'.",
            },
            {
                "original": "o que ela gosta",
                "correction": "do que ela gosta",
                "error_type": "grammar",
                "explanation": "The verb 'gostar' requires the preposition 'de'.",
            },
        ],
        "difficulty": "B1",
    }

    result = await _run_mocked_feedback(
        mock_response,
        "Eu quero comprar um prezente para minha irma, mas nao sei o que ela gosta.",
        "Portuguese",
        "English",
    )

    assert result.is_correct is False
    assert len(result.errors) == 2
    error_types = {e.error_type for e in result.errors}
    assert "spelling" in error_types
    assert "grammar" in error_types


# ---------------------------------------------------------------------------
# 5. Italian word order error
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_italian_word_order():
    mock_response = {
        "corrected_sentence": "Ho mangiato la pizza ieri.",
        "is_correct": False,
        "errors": [
            {
                "original": "Mangiato ho",
                "correction": "Ho mangiato",
                "error_type": "word_order",
                "explanation": "In Italian, the auxiliary verb 'ho' comes before the past participle 'mangiato'.",
            }
        ],
        "difficulty": "A2",
    }

    result = await _run_mocked_feedback(
        mock_response, "Mangiato ho la pizza ieri.", "Italian", "English"
    )

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "word_order"


# ---------------------------------------------------------------------------
# 6. Spanish missing word (missing preposition)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_spanish_missing_word():
    mock_response = {
        "corrected_sentence": "Voy a la tienda.",
        "is_correct": False,
        "errors": [
            {
                "original": "Voy la",
                "correction": "Voy a la",
                "error_type": "missing_word",
                "explanation": "The verb 'ir' requires the preposition 'a' before the destination.",
            }
        ],
        "difficulty": "A1",
    }

    result = await _run_mocked_feedback(
        mock_response, "Voy la tienda.", "Spanish", "English"
    )

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "missing_word"


# ---------------------------------------------------------------------------
# 7. German extra word (duplicated word)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_german_extra_word():
    mock_response = {
        "corrected_sentence": "Ich bin muede.",
        "is_correct": False,
        "errors": [
            {
                "original": "bin bin",
                "correction": "bin",
                "error_type": "extra_word",
                "explanation": "The verb 'bin' is duplicated. Remove the extra one.",
            }
        ],
        "difficulty": "A1",
    }

    result = await _run_mocked_feedback(
        mock_response, "Ich bin bin muede.", "German", "English"
    )

    assert result.is_correct is False
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "extra_word"


# ---------------------------------------------------------------------------
# 8. Correct Spanish sentence (second correct-sentence edge case)
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_correct_spanish_sentence():
    sentence = "Me gusta mucho la comida mexicana."
    mock_response = {
        "corrected_sentence": sentence,
        "is_correct": True,
        "errors": [],
        "difficulty": "A2",
    }

    result = await _run_mocked_feedback(mock_response, sentence, "Spanish", "English")

    assert result.is_correct is True
    assert result.errors == []
    assert result.corrected_sentence == sentence


# ---------------------------------------------------------------------------
# 9. Explanation language field preserved through parsing
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_explanation_field_preserved():
    expected_explanation = "You mixed two verb forms. Use 'fui' for past tense of 'ir'."
    mock_response = {
        "corrected_sentence": "Yo fui al mercado ayer.",
        "is_correct": False,
        "errors": [
            {
                "original": "soy fue",
                "correction": "fui",
                "error_type": "conjugation",
                "explanation": expected_explanation,
            }
        ],
        "difficulty": "A2",
    }

    result = await _run_mocked_feedback(
        mock_response, "Yo soy fue al mercado ayer.", "Spanish", "English"
    )

    assert len(result.errors) == 1
    assert result.errors[0].explanation == expected_explanation
