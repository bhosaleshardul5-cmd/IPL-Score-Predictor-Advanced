
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "processed" / "cricsheet_ipl.csv"

df = pd.read_csv(DATA_PATH, low_memory=False)

print("=" * 65)
print("IPL DATA QUALITY AND FEATURE INVESTIGATION")
print("=" * 65)

# --------------------------------------------------
# 1. Check match identifiers and innings length
# --------------------------------------------------

print("\n1. MATCH IDENTIFIER")
print("Match identifier column: mid")
print("Unique matches:", df["mid"].nunique())

print("\n2. OVERS DISTRIBUTION")
print("Maximum overs:", df["overs"].max())

print("\nRows with overs above 20:")
print(
    df.loc[
        df["overs"] > 20,
        ["mid", "date", "batting_team", "overs", "runs", "total"]
    ].head(10).to_string(index=False)
)

# --------------------------------------------------
# 2. Convert overs into legal-ball counts
#
# The converter uses decimal overs:
# 15 overs and 3 balls = 15.5 decimal overs.
# --------------------------------------------------

whole_overs = np.floor(df["overs"]).astype(int)
fractional_overs = df["overs"] - whole_overs

legal_balls = (
    whole_overs * 6
    + np.rint(fractional_overs * 6).astype(int)
)

print("\n3. LEGAL-BALL CHECK")
print("Minimum legal balls:", legal_balls.min())
print("Maximum legal balls:", legal_balls.max())

print(
    "Rows above 120 legal balls:",
    int((legal_balls > 120).sum())
)

print(
    "Rows at or below 120 legal balls:",
    int((legal_balls <= 120).sum())
)

# --------------------------------------------------
# 3. Derive candidate features
# --------------------------------------------------

df["legal_balls"] = legal_balls.clip(lower=0, upper=120)

df["remaining_balls"] = 120 - df["legal_balls"]

df["wickets_remaining"] = (10 - df["wickets"]).clip(
    lower=0,
    upper=10,
)

df["innings_phase"] = pd.cut(
    df["legal_balls"],
    bins=[-1, 36, 90, 120],
    labels=["Powerplay", "Middle overs", "Death overs"],
)

df["scoring_momentum"] = (
    df["last_five_run_rate"] - df["current_run_rate"]
)

# --------------------------------------------------
# 4. Display derived feature summaries
# --------------------------------------------------

print("\n4. CANDIDATE FEATURES")
print(
    df[
        [
            "overs",
            "legal_balls",
            "remaining_balls",
            "wickets",
            "wickets_remaining",
            "current_run_rate",
            "last_five_run_rate",
            "scoring_momentum",
            "innings_phase",
        ]
    ].describe(include="all").to_string()
)

print("\n5. INNINGS PHASE COUNTS")
print(df["innings_phase"].value_counts(dropna=False).to_string())

print("\n6. EXTREME VALUES")
for column in [
    "runs",
    "wickets",
    "overs",
    "runs_last_5",
    "current_run_rate",
    "last_five_run_rate",
    "total",
]:
    print(f"{column}: {df[column].min()} to {df[column].max()}")

print("\nInvestigation complete. No files were modified.")
