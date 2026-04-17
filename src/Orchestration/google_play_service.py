# import datetime
# from typing import List

from dataclasses import dataclass
import pandas as pd

# from src.Orchestration.path_helper import PathHelper, PipelineStep
from src.DataAccess.Ingestion.google_play_wrapper import GooglePlayScraper
from src.DataAccess.Refinary.google_review_cleaner import GoogleReviewCleaner, review_columns
# from src.DataAccess.Storage.csv_review_repository import CSVReviewRepository
from src.Models.gemini_flash_summary_model import GeminiFlashModel, Sentiments
from src.Models.nlptown_bert_base_model import BertBaseSentimentModel


# Cuántas reseñas mandarle al modelo de resumen (no saturar?)
MAX_REVIEWS_FOR_SUMMARY = 80

@dataclass
class FeatureSummary:
    positive: list[str]
    negative: list[str]


class GooglePlayService:

    def __init__(self, 
                 scraper: GooglePlayScraper,
                 cleaner: GoogleReviewCleaner,
                 #repository: CSVReviewRepository,
                 #path_helper: PathHelper,
                 sentiment_model: BertBaseSentimentModel,
                 summarization_model: GeminiFlashModel):
        print("Initializing GooglePlayService...")
        print("Scraper:", scraper)
        self.scraper = scraper
        print("Cleaner:", cleaner)
        self.cleaner = cleaner
        #self.repository = repository
        print("Sentiment Model:", sentiment_model)
        self.sentiment_model = sentiment_model
        print("Summarization Model:", summarization_model)
        self.summarization_model = summarization_model
        #self.path_helper = path_helper
    

    def run_pipeline(self, app_id: str, limit=1000) -> FeatureSummary:

        reviews_df = pd.DataFrame(self.scraper.get_reviews(app_id, limit))
        reviews_df = self.cleaner.clean_reviews(reviews_df)

        sentiments = self.sentiment_model.classify_sentiment(reviews_df[review_columns.CONTENT_COL].tolist())
        reviews_df[review_columns.SENTIMENT_COL] = sentiments

        weighted_positive_reviews = self.prepare_weighted_reviews(reviews_df, 
                                                                  Sentiments.POSITIVE.value)
        weighted_negative_reviews = self.prepare_weighted_reviews(reviews_df,
                                                                  Sentiments.NEGATIVE.value)
        
        positive_features = self.summarization_model.extract_top_features(weighted_positive_reviews,
                                                                          sentiment = Sentiments.POSITIVE.value,
                                                                     top_n=3)
        
        negative_features = self.summarization_model.extract_top_features(weighted_negative_reviews,
                                                                          sentiment = Sentiments.NEGATIVE.value,
                                                                          top_n=3)
        
        return FeatureSummary(positive=positive_features, negative=negative_features)


    def prepare_weighted_reviews(self,
                                 df: pd.DataFrame,
                                 sentiment: str) -> str:

        subset = df[df[review_columns.SENTIMENT_COL] == sentiment]

        if subset.empty:
            return ""

        if review_columns.LIKES_COL in subset.columns:
            subset = subset.sort_values(review_columns.LIKES_COL, ascending=False)

        subset = subset.head(MAX_REVIEWS_FOR_SUMMARY) # limitamos para no saturar el modelo

        lines = []
        for idx, (_, row) in enumerate(subset.iterrows()):
            text = str(row[review_columns.CONTENT_COL])[:500]
            likes = int(row.get(review_columns.LIKES_COL, 0))
            version = row.get(review_columns.VERSION_COL, "?")

            if likes > 0:
                lines.append(f"{idx+1}. [👍{likes} likes | v{version}] {text}")
            else:
                lines.append(f"{idx+1}. [v{version}] {text}")

        return "\n".join(lines)

    
    def analyze_by_version(self, df: pd.DataFrame) -> str:
        """Orquesta el análisis por versión."""
        df_v = self.cleaner.prepare_version_data(df)

        if df_v.empty:
            return "(No hay datos de versión)"

        stats = self.cleaner.group_version(df_v)
        stats = self.cleaner.filter_versions(stats)

        return self.cleaner.format_version_report(stats)
    

    ### DEPRECATED METHODS - to be removed after refactor ###
    
    # def deprecated_collect_and_store_reviews(self, 
    #                               app_id, 
    #                               limit=1000) -> pd.DataFrame:
    #     reviews_df = pd.DataFrame(self.scraper.get_reviews(app_id, limit))
    #     path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.RAW)
    #     self.repository.save(reviews_df, path)

    #     return reviews_df


    # def deprecated_clean_reviews(self, 
    #                   reviews_df: pd.DataFrame) -> pd.DataFrame:

    #     reviews_df = self.cleaner.clean_reviews(reviews_df)
    #     path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.CLEAN)
    #     self.repository.save(reviews_df, path)

    #     return reviews_df


    # def deprecated_analyze_sentiment(self, 
    #                       reviews_df: pd.DataFrame) -> pd.DataFrame:
        
    #     if not self.cleaner.validate_schema(reviews_df):
    #         raise ValueError("Invalid review data")

    #     reviews_df[review_columns.SENTIMENT_COL] = self.sentiment_model.classify_sentiment(reviews_df, content_col=review_columns.CONTENT_COL)
    #     path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.ANALYZED)
    #     self.repository.save(reviews_df, path)
    #     return reviews_df


    # def deprecated_build_summary(self, 
    #                   reviews_df: pd.DataFrame) -> pd.DataFrame:

    #     positive_features = self._summarize_reviews(reviews_df, Sentiments.POSITIVE, MAX_REVIEWS_FOR_SUMMARY)
    #     negative_features = self._summarize_reviews(reviews_df, Sentiments.NEGATIVE, MAX_REVIEWS_FOR_SUMMARY)

    #     summary_df = pd.DataFrame({
    #         "positive_features": [positive_features],
    #         "negative_features": [negative_features]
    #     })

    #     path = self.path_helper.build_review_step_filename(pipeline_step=PipelineStep.SUMMARY)

    #     self.repository.save(summary_df, path)

    #     return summary_df
    
    # def deprecated_summarize_reviews(self, 
    #                        reviews_df: pd.DataFrame,
    #                        sentiment: Sentiments,
    #                        sample_size:int,
    #                        top_n:int=3) -> List[str]:

    #     sentiment_filtered_reviews = reviews_df[reviews_df[review_columns.SENTIMENT_COL] == sentiment.value]
    #     # Muestra para no exceder el contexto del modelo
    #     sample = sentiment_filtered_reviews.head(sample_size)

    #     if sample.empty:
    #         return []   
        
    #     top_features = self.summarization_model.extract_top_features(sample[review_columns.CONTENT_COL].tolist(),
    #                                                                  sentiment = sentiment.value,
    #                                                                  top_n=top_n)
    #     return top_features
    

    # def deprecated_run_pipeline(self, app_id: str, limit=1000) -> dict[str, pd.DataFrame]:

    #     timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    #     self.path_helper.set_filepath_components(app_id, timestamp)

    #     raw_df = self.local_collect_and_store_reviews(app_id, limit)
    #     clean_df = self.local_clean_reviews(raw_df)
    #     analyzed = self.local_analyze_sentiment(clean_df)
    #     summary = self.local_build_summary(analyzed)

    #     return {
    #         "raw": raw_df,
    #         "clean": clean_df,
    #         "analyzed": analyzed,
    #         "summary": summary
    #         }