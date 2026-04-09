import pandas as pd
from transformers import pipeline

# Default HuggingFace model used for sentiment classification
SENTIMENT_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class RobertaSentimentModel:
    """
    Sentiment classification wrapper around the HuggingFace
    'cardiffnlp/twitter-roberta-base-sentiment-latest' model.

    Responsibilities
    ----------------
    - Load the sentiment model once during initialization
    - Perform batched inference on review text
    - Append a 'sentiment' column to the provided DataFrame

    """

    def __init__(self):
        """
        Initializes the HuggingFace inference pipeline.

        The model is loaded once when the object is created,
        avoiding model loading step on every inference call.
        """

        print(f"Loading model: {SENTIMENT_MODEL}")

        # HuggingFace pipeline for sentiment classification
        self.pipeline = pipeline(
            "text-classification",
            model=SENTIMENT_MODEL,
            truncation=True,  # Ensures input longer than the model context is truncated
            max_length=512,   # Maximum token length accepted by the model
        )

        # TODO:
        # Consider allowing the model name to be injected via the constructor
        # so different sentiment models can be swapped without modifying the class.

        # TODO:
        # Consider adding device configuration (CPU / GPU).
        # Example:
        # pipeline(..., device=0) for GPU
        # pipeline(..., device=-1) for CPU


    def classify_sentiment(self, df: pd.DataFrame, content_col: str) -> pd.DataFrame:
        """
        Runs sentiment classification on a DataFrame column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing review text.
        content_col : str
            Name of the column containing the text to classify.

        Returns
        -------
        pd.DataFrame
            The same DataFrame with an additional column:

            'sentiment' → predicted label (positive / neutral / negative)

        Notes
        -----
        - Inference is performed in batches to reduce memory overhead
          and improve throughput.
        - The method mutates the input DataFrame by adding a column.
        """

        # Run inference on the review column
        results = self.pipeline(
            df[content_col],
            batch_size=32  # Batch inference for efficiency
        )

        # Extract labels from model output
        # Example model output:
        # {'label': 'Positive', 'score': 0.98}
        df["sentiment"] = [
            r["label"].lower() for r in results
        ]

        return df