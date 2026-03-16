"""System prompt and LLM interaction for language feedback."""

import json

from openai import AsyncOpenAI

from app.models import FeedbackRequest, FeedbackResponse

SYSTEM_PROMPT = """\
You are a language-learning assistant. A student is practicing writing in their \
target language. Your job is to analyze their sentence, find errors, and provide \
helpful feedback.

RULES:
1. If the sentence is already correct, return is_correct=true, an empty errors \
array, and set corrected_sentence to the original sentence exactly.
2. For each error, identify the original text, provide the correction, classify \
the error type, and explain the error in the learner's NATIVE language so they \
can understand.
3. Error types must be one of: grammar, spelling, word_choice, punctuation, \
word_order, missing_word, extra_word, conjugation, gender_agreement, \
number_agreement, tone_register, other.
4. Assign a CEFR difficulty level (A1–C2) based on the complexity of the \
sentence (vocabulary, grammar structures used), NOT based on whether it has errors.
5. The corrected_sentence should be the minimal correction -- preserve the \
learner's original meaning and style as much as possible.
6. Explanations should be concise (1–2 sentences), friendly, and educational.

Respond with valid JSON matching this exact schema:
{
  "corrected_sentence": "string",
  "is_correct": boolean,
  "errors": [
    {
      "original": "string",
      "correction": "string",
      "error_type": "string",
      "explanation": "string (in native language)"
    }
  ],
  "difficulty": "A1|A2|B1|B2|C1|C2"
}
"""


async def get_feedback(request: FeedbackRequest) -> FeedbackResponse:
    client = AsyncOpenAI()

    user_message = (
        f"Target language: {request.target_language}\n"
        f"Native language: {request.native_language}\n"
        f"Sentence: {request.sentence}"
    )

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    return FeedbackResponse(**data)
