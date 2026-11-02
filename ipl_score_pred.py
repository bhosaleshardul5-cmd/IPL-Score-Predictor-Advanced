
from pathlib import Path
import json
import joblib
import pandas as pd

# Project directories
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model_bundle.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

# Features used during model training
FEATURES = [
    "batting_team",
    "bowling_team",
    "venue",
    "runs",
    "wickets",
    "overs",
    "runs_last_5",
    "wickets_last_5",
    "current_run_rate",
    "last_five_run_rate",
]

CATEGORICAL_FEATURES = [
    "batting_team",
    "bowling_team",
    "venue",
]

NUMERIC_FEATURES = [
    "runs",
    "wickets",
    "overs",
    "runs_last_5",
    "wickets_last_5",
    "current_run_rate",
    "last_five_run_rate",
]


def load_models():
    """Load the trained model bundle and its metadata."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model bundle not found: {MODEL_PATH}\n"
            "Run: python .\\train_models.py"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}"
        )

    bundle = joblib.load(MODEL_PATH)

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)

    return bundle, metadata


def _get_model(bundle, metadata, model_name):
    """Find the requested model in the saved bundle."""
    if model_name == "Best model":
        model_name = metadata.get("best_model")

    # Support bundles that store models inside a dictionary.
    if isinstance(bundle, dict):
        if "models" in bundle and isinstance(bundle["models"], dict):
            models = bundle["models"]
        else:
            models = bundle

        if model_name not in models:
            available = ", ".join(
                str(name) for name in models.keys()
            )
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {available}"
            )

        return models[model_name]

    # Support a bundle containing a single pipeline.
    return bundle


def predict_score(
    batting_team,
    bowling_team,
    runs,
    wickets,
    overs,
    runs_last_5,
    wickets_last_5,
    venue,
    model_name="Best model",
):
    """
    Predict the final innings score using the trained model.

    Parameters:
        overs: Completed overs as decimal overs.
               Example: 15 overs and 3 balls = 15.5.
        runs_last_5: Runs scored during the recent five-over window,
                     or the available innings so far if fewer than
                     five overs have been completed.
        wickets_last_5: Wickets lost during that recent window.
        venue: Match ground/venue name.
    """
    # Validate inputs
    if not 0 <= runs <= 300:
        raise ValueError("Runs must be between 0 and 300.")

    if not 0 <= wickets <= 10:
        raise ValueError("Wickets must be between 0 and 10.")

    if not 0 <= overs <= 20:
        raise ValueError("Overs must be between 0 and 20.")

    if not 0 <= runs_last_5 <= runs:
        raise ValueError(
            "Recent runs must be between 0 and the current score."
        )

    if not 0 <= wickets_last_5 <= wickets:
        raise ValueError(
            "Recent wickets must be between 0 and current wickets."
        )

    # Calculate run rates. The first row of an innings has no
    # completed balls, so use 0 for the rate when overs == 0.
    current_run_rate = runs / overs if overs > 0 else 0.0

    recent_overs = min(5.0, overs)
    last_five_run_rate = (
        runs_last_5 / recent_overs if recent_overs > 0 else 0.0
    )

    # Build one prediction row with exactly the training features.
    input_data = pd.DataFrame(
        [{
            "batting_team": batting_team,
            "bowling_team": bowling_team,
            "venue": venue,
            "runs": runs,
            "wickets": wickets,
            "overs": overs,
            "runs_last_5": runs_last_5,
            "wickets_last_5": wickets_last_5,
            "current_run_rate": current_run_rate,
            "last_five_run_rate": last_five_run_rate,
        }],
        columns=FEATURES,
    )

    bundle, metadata = load_models()
    model = _get_model(bundle, metadata, model_name)

    prediction = float(model.predict(input_data)[0])

    # A final innings score cannot be lower than the current score.
    prediction = max(float(runs), prediction)

    # Keep predictions within a reasonable innings-score range.
    prediction = min(300.0, prediction)

    return round(prediction)
