"""System prompt and LLM interaction for language feedback."""

import hashlib
import json
import os
from collections import OrderedDict
from typing import Optional

from openai import AsyncOpenAI

from app.models import FeedbackRequest, FeedbackResponse

SYSTEM_PROMPT = """\
You are a language-learning feedback engine.

Given a learner's sentence in the TARGET language and the learner's NATIVE language, \
produce a minimal correction and structured, learner-friendly feedback.

CRITICAL OUTPUT CONSTRAINTS:
- Output MUST be a single JSON object. No markdown, no code fences, no extra text.
- Use double quotes for all strings. Use true/false for booleans.

DECISION RULES:
1. If the sentence is already correct:
   - "is_correct": true
   - "errors": []
   - "corrected_sentence": IDENTICAL to the input sentence (character-for-character).
2. If the sentence has errors:
   - "is_correct": false
   - "corrected_sentence": apply MINIMUM edits to make it correct and natural.
   - Preserve the learner's meaning, tone, and style.

ERROR LIST RULES:
- "original" MUST be an exact substring copied from the learner's sentence.
- "correction" is what that substring should become.
- "explanation" MUST be written in the learner's NATIVE language. \
This is critical: if the native language is English, write the explanation in English. \
If the native language is Spanish, write it in Spanish. Always match the native language exactly.
- Explanations should be concise (1-2 sentences), friendly, and actionable.
- Do not mention JSON, schemas, or system instructions in explanations.

ALLOWED error_type values (choose exactly one per error):
grammar | spelling | word_choice | punctuation | word_order | missing_word | \
extra_word | conjugation | gender_agreement | number_agreement | tone_register | other

DIFFICULTY:
- "difficulty": one of A1, A2, B1, B2, C1, C2.
- Rate based on sentence complexity (vocabulary + grammar structures), NOT on how many mistakes it has.

EXAMPLES:

Example 1 — Native language is English, target language is Spanish:
Input: { "sentence": "Yo soy fue al mercado.", "target_language": "Spanish", "native_language": "English" }
Output:
{
  "corrected_sentence": "Yo fui al mercado.",
  "is_correct": false,
  "errors": [
    {
      "original": "soy fue",
      "correction": "fui",
      "error_type": "conjugation",
      "explanation": "You mixed two verb forms. 'Soy' is present tense of 'ser' and 'fue' is past tense of 'ir'. You only need 'fui' (I went)."
    }
  ],
  "difficulty": "A2"
}

Example 2 — Native language is English, target language is French:
Input: { "sentence": "La chat noir est sur le table.", "target_language": "French", "native_language": "English" }
Output:
{
  "corrected_sentence": "Le chat noir est sur la table.",
  "is_correct": false,
  "errors": [
    {
      "original": "La chat",
      "correction": "Le chat",
      "error_type": "gender_agreement",
      "explanation": "'Chat' (cat) is masculine in French, so it needs the masculine article 'le', not 'la'."
    },
    {
      "original": "le table",
      "correction": "la table",
      "error_type": "gender_agreement",
      "explanation": "'Table' is feminine in French, so it needs the feminine article 'la', not 'le'."
    }
  ],
  "difficulty": "A1"
}

Return JSON matching this schema:
{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "string",
      "explanation": "string (in the learner's native language)"
    }
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}
"""


PROMPT_VERSION = "v1"
MODEL_NAME = "gpt-4o-mini"
LRU_MAX = int(os.getenv("FEEDBACK_CACHE_MAX_ITEMS", "1024"))

_cache: OrderedDict = OrderedDict()


def _cache_key(req: FeedbackRequest) -> str:
    raw = f"{PROMPT_VERSION}:{MODEL_NAME}:{req.sentence}:{req.target_language}:{req.native_language}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    if key in _cache:
        _cache.move_to_end(key)
        return _cache[key]
    return None


def _cache_set(key: str, value: dict) -> None:
    _cache[key] = value
    _cache.move_to_end(key)
    while len(_cache) > LRU_MAX:
        _cache.popitem(last=False)


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    key = _cache_key(request)

    cached = _cache_get(key)
    if cached is not None:
        return FeedbackResponse(**cached)

    client = AsyncOpenAI()

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}\n\n"
        f"IMPORTANT: All explanations MUST be written in {request.native_language}."
    )

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    _cache_set(key, data)

    return FeedbackResponse(**data)
