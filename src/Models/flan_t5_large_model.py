from enum import StrEnum
import re
from transformers import pipeline

class Sentiments(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"

class FlanT5LargeModel:
    """
    Encapsulates summarization / feature extraction using the FLAN-T5 Large model.

    Responsibilities:
        - Build prompts for the model
        - Run inference
        - Parse the model output into structured features

    Non-responsibilities (handled elsewhere in the pipeline):
        - Sentiment filtering
        - Data loading
        - Data cleaning
        - Review sampling strategies
    """

    SUMMARY_MODEL = "google/flan-t5-large"

    # TODO: Move these constants to a global config module if reused across models
    MAX_REVIEWS_FOR_SUMMARY = 80
    MAX_CHARS_PER_REVIEW = 512

    def __init__(self):
        print(f"Loading model: {self.SUMMARY_MODEL}")

        self.pipeline = pipeline(
            "text2text-generation",  # TODO: verify correct pipeline type for FLAN-T5
            model=self.SUMMARY_MODEL,
            truncation=True,
            max_length=512,  # TODO: verify correct context size for the model
        )

    def extract_top_features(self, 
                             reviews: list[str],
                             sentiment: str, 
                             top_n: int) -> list[str]:
        """
        Extracts the most mentioned features from a set of reviews.

        Parameters
        ----------
        reviews : list[str]
            List of review texts already prepared upstream
            (filtered, sampled, and cleaned).
        sentiment : str
            Sentiment category (used only as contextual information in the prompt).
        top_n : int
            Number of features to extract.

        Returns
        -------
        list[str]
            List of extracted features.
        """

        if not reviews:
            return [f"(No reviews found for sentiment '{sentiment}')"]

        # NOTE:
        # The model assumes the reviews have already been sampled upstream.
        # Dataset size control (e.g., limiting to MAX_REVIEWS_FOR_SUMMARY)
        # should be handled by the orchestration/pipeline layer.

        prompt = self._build_summary_prompt(reviews, sentiment, top_n)

        output = self.pipeline(
            prompt,
            max_new_tokens=200,
            do_sample=False,
            # TODO: Evaluate batching if multiple prompts are processed
        )

        # NOTE:
        # Output format depends on pipeline type
        # TODO: Confirm correct key ("generated_text" vs "summary_text")
        raw = output[0]["generated_text"].strip()

        return self._parse_features(raw, top_n)
    


    def _build_summary_prompt(self, reviews: list[str], sentiment: str, top_n: int) -> str:
        """
        Constructs the instruction prompt for FLAN-T5.
        """

        joined = " | ".join(
            r[:self.MAX_CHARS_PER_REVIEW].replace("\n", " ")
            for r in reviews
        )

        # TODO:
        # Prompt length is currently unconstrained.
        # This may exceed the model context window.
        # Consider implementing a total character/token budget.

        return (
            f"You are analyzing user reviews of a mobile app. "
            f"Below are {sentiment} reviews separated by '|'. "
            f"List the top {top_n} most mentioned {sentiment} aspects or features "
            f"as a numbered list. Be concise.\n\n"
            f"Reviews: {joined}"
        )


    def _parse_features(self, raw_output: str, top_n: int) -> list[str]:
        """
        Parses the model output into a list of features.
        """

        # NOTE:
        # Parsing assumes numbered list format from the model.
        # TODO: Evaluate robustness of this parsing strategy.

        lines = re.split(r"\n|\d+[\.\)]", raw_output)

        features = [
            l.strip(" -–•")
            for l in lines
            if len(l.strip()) > 3
        ]

        if not features:
            return [raw_output]

        return features[:top_n]