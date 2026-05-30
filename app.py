# app.py
# Anime Recommendation AI
# Uses:
# - NLP module with Keyword Matching and TF-IDF
# - Numeric ML module with model comparison
# - Computer Vision module with CLIP image matching

import gradio as gr
import pandas as pd

from ml_model import train_numeric_models, predict_anime_score
from nlp_model import (
    calculate_keyword_score,
    calculate_tfidf_scores,
    extract_user_mood_openai,
    calculate_mood_score,
)
from cv_model import (
    calculate_clip_visual_scores,
    describe_visual_match,
)


DATA_PATH = "data/anime_data.csv"


def load_data():
    return pd.read_csv(DATA_PATH)


def recommend_anime(user_prompt, image):
    df = load_data()

    best_model, best_model_name, evaluation_results = train_numeric_models(df)

    user_mood = extract_user_mood_openai(user_prompt)
    tfidf_scores = calculate_tfidf_scores(user_prompt, df)

    clip_result = calculate_clip_visual_scores(image, df)
    visual_scores = clip_result["visual_scores"]

    question_text = str(user_prompt).lower()

    is_image_identification_question = (
        "which anime" in question_text
        or "what anime" in question_text
        or "what anime is this" in question_text
        or "which anime is this" in question_text
        or "which anime is that" in question_text
        or "what anime is that" in question_text
        or "who is this" in question_text
        or "which character" in question_text
        or "what is this" in question_text
    )

    results = []

    for index, row in df.iterrows():
        keyword_score = calculate_keyword_score(user_prompt, row)
        tfidf_score = float(tfidf_scores[index])
        mood_score = calculate_mood_score(user_mood, row)

        ml_numeric_score, predicted_anime_score = predict_anime_score(
            best_model,
            row,
        )

        visual_score = float(visual_scores[index])

        combined_nlp_score = (
            0.50 * keyword_score
            + 0.50 * tfidf_score
        )

        # If the user asks to identify the uploaded image,
        # the CLIP visual score gets a much higher weight.
        if image is not None and is_image_identification_question:
            final_score = (
                0.10 * combined_nlp_score
                + 0.10 * mood_score
                + 0.10 * ml_numeric_score
                + 0.70 * visual_score
            )
        else:
            final_score = (
                0.25 * combined_nlp_score
                + 0.25 * mood_score
                + 0.30 * ml_numeric_score
                + 0.20 * visual_score
            )

        results.append(
            {
                "Title": row["title"],
                "Genres": row["genres"],
                "Episodes": int(row["episodes"]),
                "Real Score": float(row["score"]),
                "Predicted ML Score": round(predicted_anime_score, 2),
                "Keyword Score": round(keyword_score, 3),
                "TF-IDF Score": round(tfidf_score, 3),
                "Mood Score": round(mood_score, 3),
                "ML Numeric Score": round(ml_numeric_score, 3),
                "CLIP Visual Score": round(visual_score, 3),
                "Final Score": round(final_score, 3),
                "Image Style": row["image_style"],
                "Synopsis": row["synopsis"],
            }
        )

    result_df = pd.DataFrame(results)

    result_df = result_df.sort_values(
        by="Final Score",
        ascending=False,
    ).head(5)

    visual_explanation = describe_visual_match(clip_result)

    if image is not None and is_image_identification_question:
        task_mode_text = (
            "Task mode: Image identification\n"
            "Because the user asked which anime is shown in the uploaded image, "
            "the CLIP visual score was weighted more strongly.\n\n"
        )
    else:
        task_mode_text = (
            "Task mode: Recommendation\n"
            "The final score combines NLP, mood features, numeric ML and CLIP visual similarity.\n\n"
        )

    evaluation_text = (
        f"{task_mode_text}"
        f"OpenAI NLP extraction:\n"
        f"- OpenAI used: {user_mood['openai_used']}\n"
        f"- Intent: {user_mood['intent']}\n"
        f"- Preferred genres: {user_mood['preferred_genres']}\n"
        f"- Explanation: {user_mood['openai_explanation']}\n\n"
        f"NLP comparison:\n"
        f"- Approach 1: Keyword Matching\n"
        f"- Approach 2: TF-IDF Cosine Similarity\n\n"
        f"{visual_explanation}\n\n"
        f"Numeric ML comparison:\n"
        f"- Model 1: Linear Regression\n"
        f"- Model 2: Random Forest Regression\n"
        f"- Best numeric ML model: {best_model_name}\n\n"
        f"Linear Regression:\n"
        f"- MAE: {evaluation_results['Linear Regression']['MAE']}\n"
        f"- RMSE: {evaluation_results['Linear Regression']['RMSE']}\n"
        f"- R2: {evaluation_results['Linear Regression']['R2']}\n\n"
        f"Random Forest:\n"
        f"- MAE: {evaluation_results['Random Forest']['MAE']}\n"
        f"- RMSE: {evaluation_results['Random Forest']['RMSE']}\n"
        f"- R2: {evaluation_results['Random Forest']['R2']}\n\n"
        f"The final recommendation combines NLP, mood features, numeric ML "
        f"predictions and CLIP-based Computer Vision image matching."
    )

    return result_df, evaluation_text


demo = gr.Interface(
    fn=recommend_anime,
    inputs=[
        gr.Textbox(
            label="Anime preference",
            lines=4,
            placeholder="Example: I want a dark emotional action anime with strong character development or what anime is this",
        ),
        gr.Image(
            label="Optional anime image or poster",
            type="pil",
        ),
    ],
    outputs=[
        gr.Dataframe(label="Top 5 Anime Recommendations"),
        gr.Textbox(label="NLP, ML and Computer Vision Evaluation"),
    ],
    title="Anime Recommendation AI",
    description=(
        "A multimodal anime recommendation system using NLP, numeric ML "
        "and CLIP-based Computer Vision image matching."
    ),
)


demo.launch()