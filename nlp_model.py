# nlp_model.py
# NLP block for Anime Recommendation AI
# Contains:
# - keyword-based text matching
# - TF-IDF similarity
# - rule-based mood extraction
# - OpenAI-based mood and intent extraction

import os
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def build_anime_text(anime_row):
    return (
        str(anime_row["genres"]) + " "
        + str(anime_row["synopsis"]) + " "
        + str(anime_row["image_style"])
    ).lower()


def clean_text(text):
    return (
        str(text)
        .lower()
        .replace(",", " ")
        .replace(".", " ")
        .replace("!", " ")
        .replace("?", " ")
        .replace("-", " ")
    )


def calculate_keyword_score(user_prompt, anime_row):
    user_text = clean_text(user_prompt)
    anime_text = build_anime_text(anime_row)

    keywords = user_text.split()

    stopwords = {
        "i", "want", "a", "an", "the", "with", "and", "or", "for",
        "to", "of", "in", "that", "has", "have", "is", "are",
        "anime", "show", "series", "something", "looking",
        "give", "me", "please",
    }

    keywords = [word for word in keywords if word not in stopwords]

    if not keywords:
        return 0.0

    matches = 0

    for keyword in keywords:
        if keyword in anime_text:
            matches += 1

    return matches / len(keywords)


def calculate_tfidf_scores(user_prompt, df):
    anime_texts = df.apply(build_anime_text, axis=1).tolist()
    corpus = [str(user_prompt).lower()] + anime_texts

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(corpus)

    user_vector = tfidf_matrix[0:1]
    anime_vectors = tfidf_matrix[1:]

    similarities = cosine_similarity(user_vector, anime_vectors)[0]

    return similarities


def extract_user_mood_rule_based(user_prompt):
    text = str(user_prompt).lower()

    mood = {
        "dark_tone": 0.0,
        "romance_level": 0.0,
        "action_level": 0.0,
        "emotional_level": 0.0,
        "preferred_genres": [],
        "intent": "recommendation",
        "openai_used": False,
        "openai_explanation": "Rule-based fallback was used.",
    }

    dark_keywords = [
        "dark", "psychological", "thriller", "horror", "mystery",
        "serious", "brutal", "violent", "dystopian", "sad",
    ]

    romance_keywords = [
        "romance", "romantic", "love", "relationship", "couple",
        "heartwarming",
    ]

    action_keywords = [
        "action", "fight", "fighting", "battle", "battles", "war",
        "shounen", "adventure", "power", "supernatural", "strong",
    ]

    emotional_keywords = [
        "emotional", "sad", "deep", "drama", "dramatic", "touching",
        "meaningful", "heartbreaking", "story", "character development",
    ]

    if any(keyword in text for keyword in dark_keywords):
        mood["dark_tone"] = 1.0

    if any(keyword in text for keyword in romance_keywords):
        mood["romance_level"] = 1.0

    if any(keyword in text for keyword in action_keywords):
        mood["action_level"] = 1.0

    if any(keyword in text for keyword in emotional_keywords):
        mood["emotional_level"] = 1.0

    if "what anime" in text or "which anime" in text:
        mood["intent"] = "image_identification"

    return mood


def extract_user_mood_openai(user_prompt):
    """
    Uses OpenAI to extract structured anime preference information.
    If no API key is available or the API call fails, the function falls back
    to the rule-based extractor.
    """

    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if OpenAI is None or not api_key:
        return extract_user_mood_rule_based(user_prompt)

    client = OpenAI(api_key=api_key)

    schema = {
        "type": "object",
        "properties": {
            "dark_tone": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "romance_level": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "action_level": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "emotional_level": {
                "type": "number",
                "minimum": 0,
                "maximum": 1,
            },
            "preferred_genres": {
                "type": "array",
                "items": {"type": "string"},
            },
            "intent": {
                "type": "string",
                "enum": [
                    "recommendation",
                    "image_identification",
                    "general_question",
                ],
            },
            "short_explanation": {
                "type": "string",
            },
        },
        "required": [
            "dark_tone",
            "romance_level",
            "action_level",
            "emotional_level",
            "preferred_genres",
            "intent",
            "short_explanation",
        ],
        "additionalProperties": False,
    }

    try:
        response = client.responses.create(
            model=model_name,
            input=[
                {
                    "role": "system",
                    "content": (
                        "You extract structured anime preferences from user text. "
                        "Return only values that help an anime recommendation system. "
                        "Use numbers between 0 and 1 for mood levels."
                    ),
                },
                {
                    "role": "user",
                    "content": str(user_prompt),
                },
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "anime_preference_extraction",
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        parsed = json.loads(response.output_text)

        return {
            "dark_tone": float(parsed["dark_tone"]),
            "romance_level": float(parsed["romance_level"]),
            "action_level": float(parsed["action_level"]),
            "emotional_level": float(parsed["emotional_level"]),
            "preferred_genres": parsed["preferred_genres"],
            "intent": parsed["intent"],
            "openai_used": True,
            "openai_explanation": parsed["short_explanation"],
        }

    except Exception as error:
        fallback = extract_user_mood_rule_based(user_prompt)
        fallback["openai_explanation"] = (
            f"OpenAI extraction failed, rule-based fallback was used. Error: {error}"
        )
        return fallback


def calculate_mood_score(user_mood, anime_row):
    total_difference = 0.0

    for feature in [
        "dark_tone",
        "romance_level",
        "action_level",
        "emotional_level",
    ]:
        user_value = float(user_mood[feature])
        anime_value = float(anime_row[feature])

        total_difference += abs(user_value - anime_value)

    mood_score = 1 - (total_difference / 4)

    return max(0.0, min(1.0, mood_score))