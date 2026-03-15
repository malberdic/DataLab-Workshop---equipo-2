import pandas as pd
from transformers import pipeline

SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class RobertaSentimentModel:
    def __init__(self):
        print(f"Loading model: {SENTIMENT_MODEL}")
        self.pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            truncation=True,
            max_length=512,
        )

    def classify_sentiment(self, df: pd.DataFrame, content_col: str):

        results = self.pipeline(df[content_col], batch_size=32)
    
        df["sentiment"] = [r["label"].lower() for r in results]
    
        return df