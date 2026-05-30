# app.py
# Anime Recommendation AI
# Erste funktionierende Version:
# - lädt Anime-Daten aus CSV
# - analysiert den User-Wunsch textbasiert
# - berechnet einfache Recommendation Scores
# - gibt Top-3 Anime-Empfehlungen aus

import gradio as gr
import pandas as pd


DATA_PATH = "data/anime_sample.csv"


def load_data():
    """
    Lädt die Anime-Daten aus der CSV-Datei.
    """
    return pd.read_csv(DATA_PATH)


def calculate_text_score(user_prompt, anime_row):
    """
    Berechnet einen einfachen Text-Score zwischen User-Wunsch
    und Anime-Genres/Synopsis.
    """
    user_text = str(user_prompt).lower()
    anime_text = (
        str(anime_row["genres"]) + " " + str(anime_row["synopsis"])
    ).lower()

    keywords = user_text.replace(",", " ").replace(".", " ").split()

    if not keywords:
        return 0

    matches = 0

    for keyword in keywords:
        if keyword in anime_text:
            matches += 1

    return matches / len(keywords)


def calculate_numeric_score(anime_row):
    """
    Berechnet einen einfachen Numeric Score aus Score und Members.
    Dies ist ein erster Platzhalter für den späteren ML Numeric Block.
    """
    score = float(anime_row["score"])
    members = float(anime_row["members"])

    # Score normalisieren: 0 bis 1
    normalized_score = score / 10

    # Members grob normalisieren
    normalized_members = min(members / 4_000_000, 1)

    numeric_score = 0.7 * normalized_score + 0.3 * normalized_members

    return numeric_score


def recommend_anime(user_prompt):
    """
    Erstellt Top-3 Anime-Empfehlungen.
    Kombiniert aktuell:
    - einfachen Text-Score
    - einfachen Numeric Score

    Später erweitern wir das mit:
    - ML-Modell
    - Computer Vision
    - LLM-Erklärung
    """
    df = load_data()

    results = []

    for _, row in df.iterrows():
        text_score = calculate_text_score(user_prompt, row)
        numeric_score = calculate_numeric_score(row)

        final_score = 0.6 * text_score + 0.4 * numeric_score

        results.append(
            {
                "Title": row["title"],
                "Genres": row["genres"],
                "Episodes": row["episodes"],
                "Anime Score": row["score"],
                "Recommendation Score": round(final_score, 3),
                "Synopsis": row["synopsis"],
            }
        )

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by="Recommendation Score",
        ascending=False,
    ).head(3)

    explanation = (
        "The recommendations are based on a first combination of text matching "
        "between the user preference and the anime descriptions, together with "
        "numeric popularity and rating features. This is the first baseline version "
        "and will later be extended with ML, NLP and Computer Vision."
    )

    return result_df, explanation


demo = gr.Interface(
    fn=recommend_anime,
    inputs=[
        gr.Textbox(
            label="Anime preference",
            lines=4,
            placeholder="Example: I want an emotional anime with action, drama and a strong story.",
        ),
    ],
    outputs=[
        gr.Dataframe(label="Top 3 Anime Recommendations"),
        gr.Textbox(label="Explanation"),
    ],
    title="Anime Recommendation AI",
    description=(
        "A first baseline version of a multimodal anime recommendation system. "
        "The final project will combine NLP, Computer Vision and Numeric ML."
    ),
)


demo.launch()
