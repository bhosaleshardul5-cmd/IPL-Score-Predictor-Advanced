
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cricsheet_ipl.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

print("Loading IPL dataset...")
df = pd.read_csv(DATA_PATH, low_memory=False)

print("\n" + "=" * 60)
print("1. DATASET OVERVIEW")
print("=" * 60)

print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print(f"Matches: {df['match_id'].nunique():,}" if "match_id" in df.columns else "Match ID column not found")
print(f"Columns available: {df.columns.tolist()}")

print("\n" + "=" * 60)
print("2. MISSING VALUES")
print("=" * 60)

print(df.isnull().sum().sort_values(ascending=False).to_string())

print("\n" + "=" * 60)
print("3. NUMERIC FEATURE SUMMARY")
print("=" * 60)

numeric_columns = [
    col for col in [
        "runs",
        "wickets",
        "overs",
        "runs_last_5",
        "wickets_last_5",
        "current_run_rate",
        "last_five_run_rate",
        "total",
    ]
    if col in df.columns
]

if numeric_columns:
    print(df[numeric_columns].describe().round(2).to_string())

print("\n" + "=" * 60)
print("4. MATCH AND INNINGS IDENTIFIERS")
print("=" * 60)

for column in [
    "match_id",
    "date",
    "inning",
    "batting_team",
    "bowling_team",
    "venue",
]:
    if column in df.columns:
        print(f"\n{column}:")
        print(f"  Unique values: {df[column].nunique(dropna=True)}")
        print(f"  Missing values: {df[column].isnull().sum()}")
        print(f"  Examples: {df[column].dropna().unique()[:8].tolist()}")

print("\n" + "=" * 60)
print("5. TARGET DISTRIBUTION")
print("=" * 60)

if "total" in df.columns:
    print(df["total"].describe().round(2).to_string())
    print("\nScores above 250:", int((df["total"] > 250).sum()))
    print("Scores below 100:", int((df["total"] < 100).sum()))

print("\nDataset inspection complete.")
