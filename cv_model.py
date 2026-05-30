# cv_model.py
# Computer Vision block for Anime Recommendation AI
# Uses CLIP zero-shot image-to-text similarity
# to compare an uploaded anime image with anime titles from the dataset.

import numpy as np
import torch

from transformers import CLIPProcessor, CLIPModel


MODEL_NAME = "openai/clip-vit-base-patch32"


_clip_model = None
_clip_processor = None


def load_clip_model():
    """
    Loads CLIP model and processor.
    The model is cached globally so it is not loaded again for every request.
    """

    global _clip_model
    global _clip_processor

    if _clip_model is None or _clip_processor is None:
        _clip_model = CLIPModel.from_pretrained(MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        _clip_model.eval()

    return _clip_model, _clip_processor


def build_clip_text_prompts(df):
    """
    Creates text prompts for CLIP based on anime metadata.
    """

    prompts = []

    for _, row in df.iterrows():
        prompt = (
            f"anime poster of {row['title']}. "
            f"Genres: {row['genres']}. "
            f"Visual style: {row['image_style']}."
        )
        prompts.append(prompt)

    return prompts


def calculate_clip_visual_scores(image, df):
    """
    Compares uploaded image with all anime entries using CLIP.

    Returns:
    - visual_scores: one score per anime
    - best_match_title: anime title with highest visual similarity
    - best_match_score: highest visual score
    """

    if image is None:
        neutral_scores = np.full(len(df), 0.5)

        return {
            "has_image": False,
            "visual_scores": neutral_scores,
            "best_match_title": "No image uploaded",
            "best_match_score": 0.5,
        }

    model, processor = load_clip_model()

    image = image.convert("RGB")
    text_prompts = build_clip_text_prompts(df)

    inputs = processor(
        text=text_prompts,
        images=image,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probabilities = logits_per_image.softmax(dim=1).cpu().numpy()[0]

    best_index = int(np.argmax(probabilities))
    best_match_title = str(df.iloc[best_index]["title"])
    best_match_score = float(probabilities[best_index])

    return {
        "has_image": True,
        "visual_scores": probabilities,
        "best_match_title": best_match_title,
        "best_match_score": best_match_score,
    }


def describe_visual_match(clip_result):
    """
    Creates a short explanation for the CV result.
    """

    if not clip_result["has_image"]:
        return "No image was uploaded. The visual score was kept neutral."

    return (
        f"Computer Vision result:\n"
        f"- Model: CLIP zero-shot image-to-text similarity\n"
        f"- Closest visual anime match: {clip_result['best_match_title']}\n"
        f"- Visual match confidence: {round(clip_result['best_match_score'], 3)}"
    )