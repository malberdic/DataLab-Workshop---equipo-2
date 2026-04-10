from enum import StrEnum, auto
from pathlib import Path

# Go up to project root
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

# Ensure directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)


class PipelineStep(StrEnum):
    RAW: str = auto()
    CLEAN: str = auto()
    ANALYZED: str = auto()
    SUMMARY: str = auto()

class PathHelper:
    def __init__(self):
        self.app_id = None
        self.timestamp = None

    def build_review_step_filename(self,  
                                   pipeline_step : PipelineStep) -> Path:
        if self.timestamp is None or self.app_id is None:
            raise RuntimeError("PathHelper not initialized. Call set_filepath_components first.")

        filename = f"{self.app_id}_{self.timestamp}_{pipeline_step}.csv"
        return DATA_DIR / filename
    
    def set_filepath_components(self,
                                app_id : str, 
                                timestamp : str):
        if not app_id or not timestamp:
            raise ValueError("app_id and timestamp must be provided")
        self.app_id = app_id
        self.timestamp = timestamp