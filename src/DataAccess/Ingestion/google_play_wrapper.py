print("Importing typing")
from typing import Dict, List, Optional
print("Importing google_play_scraper")
from google_play_scraper import reviews, Sort

class GooglePlayScraper:

    def __init__(self, lang: str = "es", country: str = "ar"):
        print("Initializing GooglePlayScraper with lang =", lang, "and country =", country)
        self.lang = lang
        self.country = country
        print(f"GooglePlayScraper initialized with lang='{self.lang}' and country='{self.country}'")

    def get_reviews(
        self,
        app_id: str,
        limit: int = 1000,
        sort: Sort = Sort.NEWEST
    ) -> List[Dict]:

        collected: List[Dict] = []
        token: Optional[str] = None
        remaining = limit

        while remaining > 0:

            batch, token = reviews(
                app_id,
                lang=self.lang,
                country=self.country,
                sort=sort,
                count=min(200, remaining),
                continuation_token=token
            )

            # Early exit if API returns nothing
            if not batch:
                break

            collected.extend(batch)
            remaining -= len(batch)

            # Early exit if no continuation token
            if token is None:
                break

        return collected