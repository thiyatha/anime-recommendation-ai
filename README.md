---
title: Anime Recommendation AI
emoji: 🎌
colorFrom: purple
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
---


# 🎌 Anime Recommendation AI

Anime Recommendation AI is a multimodal Gradio application that recommends anime based on user preferences, structured anime metadata, and optional image input.

The project combines three AI blocks:

- **NLP** for understanding natural language user preferences
- **ML Numeric Data** for score prediction using structured anime metadata
- **Computer Vision** for optional image-based anime matching using CLIP

The final recommendation is created by combining text similarity, mood matching, numeric ML predictions, and optional visual similarity into one recommendation score.

---

## Deployment

Hugging Face Space:

https://huggingface.co/spaces/thajee/anime-recommendation-ai

GitHub Repository:

https://github.com/thiyatha/anime-recommendation-ai

---

## AI Blocks

### NLP

The NLP block analyzes the user's text input using:

- Keyword Matching
- TF-IDF Cosine Similarity
- OpenAI-based structured preference extraction

### ML Numeric Data

The numeric ML block compares:

- Linear Regression
- Random Forest Regression

The models use structured features such as episodes, members, score values, and mood features.

### Computer Vision

The Computer Vision block uses CLIP to compare an uploaded anime image with anime titles and metadata.

---

## Data Sources

The project uses multiple data sources:

- Local anime CSV dataset
- Jikan API anime metadata
- User text input
- Optional uploaded anime image

---

## Run Locally

```bash
git clone https://github.com/thiyatha/anime-recommendation-ai.git
cd anime-recommendation-ai
pip install -r requirements.txt
python app.py
```

---

## Documentation

The full project documentation is available in:

```text
documentation.md
```
