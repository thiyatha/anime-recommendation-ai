# ml_model.py
# Numeric ML block for Anime Recommendation AI
# Trains and compares Linear Regression and Random Forest Regression

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


FEATURE_COLUMNS = [
    "episodes",
    "members",
    "dark_tone",
    "romance_level",
    "action_level",
    "emotional_level",
]

TARGET_COLUMN = "score"


def train_numeric_models(df):
    """
    Trains and compares two numeric ML models:
    - Linear Regression
    - Random Forest Regression

    Target:
    - anime score

    Features:
    - episodes
    - members
    - dark_tone
    - romance_level
    - action_level
    - emotional_level
    """

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
        ),
    }

    evaluation_results = {}
    trained_models = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        evaluation_results[model_name] = {
            "MAE": round(mae, 3),
            "RMSE": round(rmse, 3),
            "R2": round(r2, 3),
        }

        trained_models[model_name] = model

    best_model_name = min(
        evaluation_results,
        key=lambda name: evaluation_results[name]["MAE"],
    )

    best_model = trained_models[best_model_name]

    return best_model, best_model_name, evaluation_results


def predict_anime_score(model, anime_row):
    """
    Predicts the anime score with the selected best ML model.
    The predicted score is also normalized from 0-10 to 0-1.
    """

    feature_values = pd.DataFrame(
        [[anime_row[col] for col in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )

    predicted_score = model.predict(feature_values)[0]
    normalized_prediction = predicted_score / 10

    normalized_prediction = max(0.0, min(1.0, normalized_prediction))

    return normalized_prediction, predicted_score