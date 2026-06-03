# api_data.py
# External data source for Anime Recommendation AI
# The Jikan API is used only to enrich the final Top 5 recommendations.

from functools import lru_cache
import time
import requests


JIKAN_SEARCH_URL = "https://api.jikan.moe/v4/anime"


@lru_cache(maxsize=128)
def fetch_anime_metadata(title: str) -> dict:
    """
    Fetch external anime metadata from the Jikan API.

    The API is only used after the recommendation ranking.
    The core recommendation logic still works without the API.
    """

    params = {
        "q": title,
        "limit": 1,
        "sfw": True,
    }

    try:
        # Small delay to avoid too many API requests too quickly
        time.sleep(0.4)

        response = requests.get(
            JIKAN_SEARCH_URL,
            params=params,
            timeout=10,
        )

        if response.status_code == 429:
            return {}

        response.raise_for_status()
        data = response.json()

        if not data.get("data"):
            return {}

        anime = data["data"][0]

        return {
            "external_title": anime.get("title"),
            "external_score": anime.get("score"),
            "external_episodes": anime.get("episodes"),
            "external_popularity": anime.get("popularity"),
            "external_members": anime.get("members"),
            "external_synopsis": anime.get("synopsis"),
            "external_url": anime.get("url"),
            "external_image_url": anime.get("images", {})
            .get("jpg", {})
            .get("image_url"),
            "external_genres": ", ".join(
                genre.get("name", "")
                for genre in anime.get("genres", [])
            ),
        }

    except requests.RequestException:
        return {}
