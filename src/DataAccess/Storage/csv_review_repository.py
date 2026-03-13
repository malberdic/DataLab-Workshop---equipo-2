import pandas as pd


class CSVReviewRepository:

    def save(self, reviews: pd.DataFrame, path: str):
        reviews.to_csv(path, index=False)

    def load(self, path: str) -> pd.DataFrame:
        return pd.read_csv(path)