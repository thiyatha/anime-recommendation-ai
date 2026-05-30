---
title: Anime Recommendation AI
emoji: 🎌
colorFrom: purple
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---

# Apartment Price Prediction Zurich

## Project Overview

This project predicts apartment rental prices in the canton of Zurich using machine learning.  
The goal is to estimate the monthly rent of an apartment based on important apartment features.

## Objective

The objective of this project is to build a regression model that can predict apartment rental prices in the canton of Zurich.  
The solution includes:

- a trained regression model
- a newly engineered feature
- a deployed web application
- a documented iterative modeling process

## Dataset

The dataset contains apartment-related information used to predict the monthly rental price.  
The target variable is:

- `rent`

Examples of input features used in the project:

- `living_space`
- `rooms`
- `floor`
- `year_built`
- `zip_code`
- `balcony`
- `parking`

## New Feature

A new engineered feature was added to improve the model:

- `apartment_age = 2026 - year_built`

This feature represents the age of the apartment and helps the model better capture the relationship between building age and rental price.

## Data Preprocessing

The following preprocessing steps were applied:

- selected relevant input features
- handled missing numerical values
- handled missing categorical values
- encoded categorical variables
- scaled numerical variables where necessary
- created the new feature `apartment_age`
- prepared the data for cross-validation and model training

## Iterative Modeling Process

At least two modeling iterations were performed to compare different preprocessing strategies and regression models.

| Iteration | Objective | Changes Compared to Previous Iteration | Preprocessing Steps | Models Used | Hyperparameters | Cross-Validation Metrics | Decision |
|---|---|---|---|---|---|---|---|
| 1 | Build a baseline model | Initial baseline using standard apartment features | Missing value handling, categorical encoding, numerical scaling | Linear Regression, Random Forest Regressor | Linear Regression: default; Random Forest: n_estimators=200, random_state=42 | Linear Regression: RMSE = add_your_result, MAE = add_your_result, R² = add_your_result; Random Forest: RMSE = add_your_result, MAE = add_your_result, R² = add_your_result | Random Forest performed better in the baseline iteration |
| 2 | Improve model performance | Added the engineered feature `apartment_age` and improved the feature set | Missing value handling, feature engineering, encoding, scaling | Random Forest Regressor, Gradient Boosting Regressor | Random Forest: n_estimators=300, random_state=42; Gradient Boosting: n_estimators=300, learning_rate=0.05, max_depth=3, random_state=42 | Random Forest: RMSE = add_your_result, MAE = add_your_result, R² = add_your_result; Gradient Boosting: RMSE = add_your_result, MAE = add_your_result, R² = add_your_result | Gradient Boosting was selected as the final model based on the best overall performance |

## Models Used

The following regression models were tested during the project:

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Evaluation Method

The models were evaluated using cross-validation for a regression task.  
The following evaluation metrics were used:

- RMSE
- MAE
- R²

These metrics were used to compare model performance across iterations and to select the final model.

## Final Selected Model

The final selected model is:

- `Gradient Boosting Regressor`

This model was selected because it achieved the best cross-validation performance among the tested models after adding the new engineered feature.

## Application

The model is deployed as a web application on Hugging Face Spaces.

**Public application link:**  
Add your Hugging Face Space link here

## Repository Files

- `app.py` → web application
- `requirements.txt` → required Python packages
- `README.md` → project documentation
- `best_model.joblib` → saved trained model

## Conclusion

This project demonstrates a complete machine learning workflow for apartment rent prediction in the canton of Zurich.  
It includes data preprocessing, feature engineering, iterative model comparison, final model selection, and deployment as a public application.
