import datetime
from typing import List

import pandas as pd

from Orchestration.path_helper import PathHelper, PipelineStep
from DataAccess.Ingestion.google_play_wrapper import GooglePlayScraper
from DataAccess.Refinary.google_review_cleaner import GoogleReviewCleaner
from DataAccess.Storage.csv_review_repository import CSVReviewRepository
from Models.flan_t5_large_model import FlanT5LargeModel, Sentiments
from Models.roberta_sentiment_model import RobertaSentimentModel


# Cuántas reseñas mandarle al modelo de resumen (no saturar?)
MAX_REVIEWS_FOR_SUMMARY = 80
CONTENT_COL = "content"
SENTIMENT_COL = "sentiment"

class GooglePlayService:

    def __init__(self, 
                 lang="es", 
                 country="ar"):
        self.scraper = GooglePlayScraper(lang, country)
        self.cleaner = GoogleReviewCleaner()
        self.repository = CSVReviewRepository()
        self.sentiment_model = RobertaSentimentModel()
        self.summarization_model = FlanT5LargeModel()
        self.path_helper = PathHelper()

    def collect_and_store_reviews(self, 
                                  app_id, 
                                  limit=1000) -> pd.DataFrame:
        reviews_df = pd.DataFrame(self.scraper.get_reviews(app_id, limit))
        path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.RAW)
        self.repository.save(reviews_df, path)

        return reviews_df


    def clean_reviews(self, 
                      reviews_df: pd.DataFrame) -> pd.DataFrame:

        reviews_df = self.cleaner.clean_reviews(reviews_df)
        path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.CLEAN)
        self.repository.save(reviews_df, path)

        return reviews_df


    def analyze_sentiment(self, 
                          reviews_df: pd.DataFrame) -> pd.DataFrame:
        
        if not self.cleaner.validate_schema(reviews_df):
            raise ValueError("Invalid review data")

        reviews_df = self.sentiment_model.classify_sentiment(reviews_df, content_col=CONTENT_COL)
        path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.ANALYZED)
        self.repository.save(reviews_df, path)
        return reviews_df


    def build_summary(self, 
                      reviews_df: pd.DataFrame) -> pd.DataFrame:

        positive_features = self._summarize_reviews(reviews_df, Sentiments.POSITIVE, MAX_REVIEWS_FOR_SUMMARY)
        negative_features = self._summarize_reviews(reviews_df, Sentiments.NEGATIVE, MAX_REVIEWS_FOR_SUMMARY)

        summary_df = pd.DataFrame({
            "positive_features": [positive_features],
            "negative_features": [negative_features]
        })

        path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.SUMMARY)

        self.repository.save(summary_df, path)

        return summary_df
    
    def _summarize_reviews(self, 
                           reviews_df: pd.DataFrame,
                           sentiment: Sentiments,
                           sample_size:int,
                           top_n:int=3) -> List[str]:

        sentiment_filtered_reviews = reviews_df[reviews_df[SENTIMENT_COL] == sentiment.value]
        # Muestra para no exceder el contexto del modelo
        sample = sentiment_filtered_reviews.head(sample_size)

        if sample.empty:
            return []   
        
        top_features = self.summarization_model.extract_top_features(sample[CONTENT_COL].tolist(),
                                                                     sentiment = sentiment.value,
                                                                     top_n=top_n)
        return top_features
    

    def run_pipeline(self, app_id: str, limit=1000) -> dict[str, pd.DataFrame]:

        timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

        self.path_helper.set_filepath_components(app_id, timestamp)

        raw_df = self.collect_and_store_reviews(app_id, limit)
        clean_df = self.clean_reviews(raw_df)
        analyzed = self.analyze_sentiment(clean_df)
        summary = self.build_summary(analyzed)

        return {
            "raw": raw_df,
            "clean": clean_df,
            "analyzed": analyzed,
            "summary": summary
            }