"""Train and evaluate IPL final-score regression models."""
from pathlib import Path
import json
import re
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
FEATURES = ["batting_team", "bowling_team", "runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]
TARGET = "total"

ALIASES = {
    "batting_team": ["batting_team", "battingteam", "batting team", "bat_team"],
    "bowling_team": ["bowling_team", "bowlingteam", "bowling team", "bowl_team"],
    "runs": ["runs", "current_score", "current runs", "score"],
    "wickets": ["wickets", "wickets_fallen", "wickets fallen"],
    "overs": ["overs", "over", "overs_completed"],
    "runs_last_5": ["runs_last_5", "last_five_runs", "last 5 overs runs", "runs in last 5 overs"],
    "wickets_last_5": ["wickets_last_5", "last_five_wickets", "last 5 overs wickets", "wickets in last 5 overs"],
    "total": ["total", "final_score", "total_runs", "target_score", "final score"],
}

def norm(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")

def find_dataset():
    candidates = [ROOT / "ipl.csv", ROOT / "ipl_data.csv", ROOT / "data" / "ipl.csv", ROOT / "data" / "ipl_data.csv"]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Dataset not found. Put your CSV at ipl.csv or data/ipl.csv.")

def standardize_columns(df):
    lookup = {norm(c): c for c in df.columns}
    rename = {}
    for target, options in ALIASES.items():
        for option in options:
            key = norm(option)
            if key in lookup:
                rename[lookup[key]] = target
                break
    return df.rename(columns=rename)

def main():
    path = find_dataset()
    print(f"Loading dataset: {path}")
    df = standardize_columns(pd.read_csv(path))
    print("Columns detected:", list(df.columns))
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        print("\nERROR: This dataset does not contain the columns required for score prediction.")
        print("Missing:", missing)
        print("Expected equivalents for: batting team, bowling team, runs, wickets, overs,")
        print("runs in last five overs, wickets in last five overs, and final score.")
        print("If your dataset has different columns, send me the column list and we'll adapt the code.")
        sys.exit(1)

    df = df[FEATURES + [TARGET]].copy()
    for col in ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5", TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET, "runs", "wickets", "overs"])
    df = df[(df["overs"] >= 0) & (df["overs"] <= 20) & (df["wickets"] >= 0) & (df["wickets"] <= 10)]
    df["batting_team"] = df["batting_team"].fillna("Unknown").astype(str).str.strip()
    df["bowling_team"] = df["bowling_team"].fillna("Unknown").astype(str).str.strip()
    for col in ["runs_last_5", "wickets_last_5"]:
        df[col] = df[col].fillna(0)
    df = df[df["total"] >= df["runs"]]
    if len(df) < 30:
        raise ValueError(f"Only {len(df)} usable rows found; at least 30 are needed for a basic train/test split.")
    print(f"Usable rows: {len(df):,}")
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    categorical = ["batting_team", "bowling_team"]
    numeric = ["runs", "wickets", "overs", "runs_last_5", "wickets_last_5"]
    prep = ColumnTransformer([
        ("teams", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical),
        ("numbers", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric)
    ])
    candidates = {
        "Random Forest": RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.08, l2_regularization=1.0, random_state=42),
    }
    bundle, scores = {}, {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocess", prep), ("model", estimator)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
        mae = float(mean_absolute_error(y_test, pred))
        r2 = float(r2_score(y_test, pred))
        bundle[name] = pipe
        scores[name] = {"MAE_runs": round(mae, 3), "RMSE_runs": round(rmse, 3), "R2": round(r2, 4)}
        print(f"{name:24s} MAE={mae:.2f} runs | RMSE={rmse:.2f} runs | R²={r2:.3f}")
    best = min(scores, key=lambda name: scores[name]["MAE_runs"])
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "model_bundle.joblib")
    metadata = {
        "best_model": best,
        "scores": scores,
        "features": FEATURES,
        "target": TARGET,
        "rows_used": int(len(df)),
        "dataset": path.name,
        "note": "Baseline random split; for stronger validation, split by match or season."
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nBest model by test MAE: {best}")
    print(f"Saved model bundle and metadata in: {MODEL_DIR}")

if __name__ == "__main__":
    main()
