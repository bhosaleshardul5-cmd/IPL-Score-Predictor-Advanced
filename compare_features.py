
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cricsheet_ipl.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Keep rows with the fields needed for a fair comparison.
required = [
    "mid", "date", "batting_team", "bowling_team", "venue",
    "runs", "wickets", "overs", "runs_last_5", "wickets_last_5",
    "current_run_rate", "last_five_run_rate", "total",
]
df = df.dropna(subset=required).copy()

# Convert decimal overs into legal-ball counts.
whole_overs = np.floor(df["overs"]).astype(int)
fraction = df["overs"] - whole_overs
df["legal_balls"] = (
    whole_overs * 6 + np.rint(fraction * 6).astype(int)
)

# Keep innings states within the normal 20-over limit.
# This excludes only records that exceed 120 legal balls.
df = df[df["legal_balls"].between(6, 120)].copy()

# Candidate features.
df["remaining_balls"] = 120 - df["legal_balls"]
df["wickets_remaining"] = (10 - df["wickets"]).clip(0, 10)
df["scoring_momentum"] = (
    df["last_five_run_rate"] - df["current_run_rate"]
)
df["innings_phase"] = pd.cut(
    df["legal_balls"],
    bins=[5, 36, 90, 120],
    labels=["Powerplay", "Middle overs", "Death overs"],
    include_lowest=True,
).astype(str)

baseline_features = [
    "batting_team", "bowling_team", "venue", "runs", "wickets",
    "overs", "runs_last_5", "wickets_last_5",
    "current_run_rate", "last_five_run_rate",
]

extra_features = [
    "remaining_balls",
    "wickets_remaining",
    "innings_phase",
    "scoring_momentum",
]

target = "total"

# Split by whole matches, ordered by each match's date.
match_dates = (
    df.groupby("mid")["date"]
    .min()
    .sort_values(kind="stable")
)
ordered_matches = match_dates.index.tolist()
split_index = int(len(ordered_matches) * 0.80)

train_ids = set(ordered_matches[:split_index])
test_ids = set(ordered_matches[split_index:])

train = df[df["mid"].isin(train_ids)].copy()
# Use a reproducible sample for a faster experiment.
train = train.sample(
    n=min(80000, len(train)),
    random_state=42
).copy()
test = df[df["mid"].isin(test_ids)].copy()

print(f"Usable rows: {len(df):,}")
print(f"Usable matches: {df['mid'].nunique():,}")
print(f"Training rows: {len(train):,}")
print(f"Test rows: {len(test):,}")
print(f"Test starts: {test['date'].min().date()}")

categorical_base = ["batting_team", "bowling_team", "venue"]
categorical_extra = categorical_base + ["innings_phase"]

numeric_base = [
    "runs", "wickets", "overs", "runs_last_5", "wickets_last_5",
    "current_run_rate", "last_five_run_rate",
]
numeric_extra = numeric_base + [
    "remaining_balls", "wickets_remaining", "scoring_momentum",
]


def build_model(categorical, numeric):
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric,
            ),
        ]
    )

    return Pipeline([
        ("preprocessor", preprocessing),
        ("model", ExtraTreesRegressor(
            n_estimators=80,
            min_samples_leaf=2,
            max_features=0.9,
            n_jobs=2,
            random_state=42,
        )),
    ])


experiments = [
    ("Baseline - 10 features", baseline_features, categorical_base, numeric_base),
    (
        "Enhanced - 14 features",
        baseline_features + extra_features,
        categorical_extra,
        numeric_extra,
    ),
]

results = []

for name, features, categorical, numeric in experiments:
    print(f"\nTraining {name}...")

    model = build_model(categorical, numeric)
    model.fit(train[features], train[target])

    predictions = model.predict(test[features])

    mae = mean_absolute_error(test[target], predictions)
    rmse = np.sqrt(mean_squared_error(test[target], predictions))
    r2 = r2_score(test[target], predictions)

    results.append({
        "Experiment": name,
        "MAE_runs": mae,
        "RMSE_runs": rmse,
        "R2": r2,
    })

    print(f"{name}: MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)
print(pd.DataFrame(results).to_string(index=False))

print("\nNo existing models or project files were overwritten.")
