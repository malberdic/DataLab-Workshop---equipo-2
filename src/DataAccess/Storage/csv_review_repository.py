from pandas import DataFrame as pdf


class CSVReviewRepository:

    def save(self, reviews: pdf, path: str):
        reviews.to_csv(path, index=False)

    def load(self, path: str) -> pdf:
        return pdf.from_csv(path)