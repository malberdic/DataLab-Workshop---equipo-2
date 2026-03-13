import pandas as pd
from transformers import pipeline

from src.Models.analyze_reviews import classify_sentiment, extract_top_features
from DataAccess.Ingestion.google_play_wrapper import GooglePlayScraper
from src.DataAccess.Storage.csv_review_repository import CSVReviewRepository

class GooglePlayService:

    def __init__(self, lang="es", country="ar"):
        self.scraper = GooglePlayScraper(lang, country)
        self.repository = CSVReviewRepository()

    def collect_and_store_reviews(self, app_id, limit=1000, path="reviews.csv"):
        reviews_df = pd.DataFrame(self.scraper.get_reviews(app_id, limit))
        self.repository.save(reviews_df, path)

    # def analyze_reviews(self, path="reviews.csv"):
    #     reviews_df = self.repository.load(path)

    #     # Agregar análisis de sentimiento
    #     reviews_df = classify_sentiment(reviews_df)

    #     # Establecer pipeline de extracción de características
    
    #     summary_pipe = pipeline(
    #         "text2text-generation",
    #         model=SUMMARY_MODEL,
    #         truncation=True,
    #         max_length=1024,
    #         )

    #     return self.analyzer.analyze(reviews_df)
    

