
"""Train IPL score prediction models using Cricsheet data."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "processed" / "cricsheet_ipl.csv"
MODEL_DIR = ROOT / "models"

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

TARGET = "total"

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


def make_pipeline(estimator):
    """Create a preprocessing and regression pipeline."""

    preprocess = ColumnTransformer([
        (
            "categories",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="most_frequent"),
                ),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False,
                    ),
                ),
            ]),
            CATEGORICAL_FEATURES,
        ),
        (
            "numbers",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                ("scale", StandardScaler()),
            ]),
            NUMERIC_FEATURES,
        ),
    ])

    return Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Run convert_cricsheet.py first."
        )

    print(f"Loading dataset: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [TARGET, "mid", "date"]
    missing = [column for column in required if column not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            "Run the updated convert_cricsheet.py first."
        )

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in NUMERIC_FEATURES + [TARGET]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(subset=[
        "mid",
        "date",
        "runs",
        "wickets",
        "overs",
        TARGET,
    ])

    # Overs are already decimal overs, not over.ball notation.
    df = df[
        (df["overs"] >= 1)
        & (df["overs"] <= 20)
        & (df["wickets"] >= 0)
        & (df["wickets"] <= 10)
        & (df["runs"] >= 0)
        & (df[TARGET] >= df["runs"])
    ].copy()

    for column in CATEGORICAL_FEATURES:
        df[column] = (
            df[column].fillna("Unknown").astype(str).str.strip()
        )
        df.loc[df[column] == "", column] = "Unknown"

    for column in [
        "runs_last_5",
        "wickets_last_5",
        "current_run_rate",
        "last_five_run_rate",
    ]:
        df[column] = df[column].fillna(0)

    if len(df) < 100:
        raise ValueError(f"Only {len(df)} usable rows remain.")

    print(f"Usable rows: {len(df):,}")
    print(f"Matches: {df['mid'].nunique():,}")
    print(
        f"Date range: {df['date'].min().date()} to "
        f"{df['date'].max().date()}"
    )

    # Split complete matches chronologically.
    match_dates = df.groupby("mid")["date"].min().sort_values()

    if len(match_dates) < 10:
        raise ValueError(
            "Not enough matches for chronological validation."
        )

    split_position = int(len(match_dates) * 0.80)
    split_position = min(
        max(split_position, 1),
        len(match_dates) - 1,
    )

    cutoff_date = match_dates.iloc[split_position]

    train_df = df[df["date"] < cutoff_date].copy()
    test_df = df[df["date"] >= cutoff_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "Chronological split produced an empty dataset."
        )

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_test = test_df[FEATURES]
    y_test = test_df[TARGET]

    print(f"\nTraining matches: {train_df['mid'].nunique():,}")
    print(f"Test matches: {test_df['mid'].nunique():,}")
    print(f"Training rows: {len(train_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Test period starts: {test_df['date'].min().date()}")

    candidates = {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            min_samples_leaf=2,
            max_features=0.9,
            random_state=42,
            n_jobs=-1,
        ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(
            max_iter=250,
            learning_rate=0.06,
            max_leaf_nodes=15,
            l2_regularization=2.0,
            random_state=42,
        ),
        "Hist Gradient Boosting - Deeper": HistGradientBoostingRegressor(
            max_iter=300,
            learning_rate=0.04,
            max_leaf_nodes=25,
            l2_regularization=3.0,
            random_state=42,
        ),
        "Hist Gradient Boosting - Faster": HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=42,
        ),
    }

    bundle = {}
    scores = {}

    for name, estimator in candidates.items():
        print(f"\nTraining {name}...")

        model = make_pipeline(estimator)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, predictions))
        rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
        r2 = float(r2_score(y_test, predictions))

        bundle[name] = model
        scores[name] = {
            "MAE_runs": round(mae, 3),
            "RMSE_runs": round(rmse, 3),
            "R2": round(r2, 4),
        }

        print(
            f"{name}: MAE={mae:.2f} | "
            f"RMSE={rmse:.2f} | R2={r2:.3f}"
        )

    best = min(
        scores,
        key=lambda name: scores[name]["MAE_runs"],
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        bundle,
        MODEL_DIR / "model_bundle.joblib",
    )

    metadata = {
        "best_model": best,
        "scores": scores,
        "features": FEATURES,
        "target": TARGET,
        "rows_used": int(len(df)),
        "matches_used": int(df["mid"].nunique()),
        "dataset": str(DATA_FILE.relative_to(ROOT)),
        "validation": (
            "Chronological split by match date; latest 20% held out"
        ),
        "training_end_date": str(train_df["date"].max().date()),
        "test_start_date": str(test_df["date"].min().date()),
        "note": (
            "Features include venue, current run rate, and last-five-over "
            "run rate. Test matches are later than training matches."
        ),
    }

    (MODEL_DIR / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nBest model:", best)
    print("Saved model bundle and metadata to:", MODEL_DIR)


if __name__ == "__main__":
    main()
