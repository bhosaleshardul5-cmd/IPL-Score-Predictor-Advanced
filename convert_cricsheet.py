
"""Convert Cricsheet IPL JSON files into a score-prediction dataset."""

import json
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "cricsheet_raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "cricsheet_ipl.csv"

TEAM_ALIASES = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiant": "Rising Pune Supergiants",
}

COLUMNS = [
    "mid",
    "date",
    "venue",
    "batting_team",
    "bowling_team",
    "batsman",
    "bowler",
    "runs",
    "wickets",
    "overs",
    "runs_last_5",
    "wickets_last_5",
    "current_run_rate",
    "last_five_run_rate",
    "striker",
    "non-striker",
    "total",
]

NON_WICKET_DISMISSALS = {
    "retired hurt",
    "obstructing the field",
}


def counted_wickets(delivery):
    """Count dismissals that count as wickets."""
    return sum(
        wicket.get("kind", "").lower() not in NON_WICKET_DISMISSALS
        for wicket in delivery.get("wickets", [])
    )


def is_legal_delivery(delivery):
    """Wides and no-balls do not count as legal balls."""
    extras = delivery.get("extras", {})
    return "wides" not in extras and "noballs" not in extras


def convert_innings(match_id, date, venue, innings, teams):
    batting_team = innings.get("team", "")
    bowling_team = next(
        (team for team in teams if team != batting_team), ""
    )

    deliveries = [
        delivery
        for over in innings.get("overs", [])
        for delivery in over.get("deliveries", [])
    ]

    # Calculate the final innings score, including penalty runs.
    final_total = sum(
        int(delivery.get("runs", {}).get("total", 0))
        for delivery in deliveries
    )

    for penalty in innings.get("penalty_runs", []):
        final_total += int(penalty.get("runs", 0))

    rows = []
    score = 0
    wickets = 0
    legal_balls = 0
    history = []

    for delivery in deliveries:
        runs_info = delivery.get("runs", {})
        delivery_runs = int(runs_info.get("total", 0))
        wicket_count = counted_wickets(delivery)
        legal = is_legal_delivery(delivery)

        score += delivery_runs
        wickets += wicket_count

        if legal:
            legal_balls += 1

        history.append({
            "legal_ball": legal_balls,
            "runs": delivery_runs,
            "wickets": wicket_count,
        })

        # Retain deliveries associated with the last 30 legal-ball
        # positions. This is an approximation of the last five overs.
        first_ball = max(1, legal_balls - 29)
        window = [
            item
            for item in history
            if item["legal_ball"] >= first_ball
        ]

        # Begin generating prediction rows after one completed over.
        if legal_balls < 6:
            continue

        completed_overs, balls_in_over = divmod(legal_balls, 6)
        overs_value = completed_overs + balls_in_over / 6.0

        runs_last_5 = sum(item["runs"] for item in window)
        wickets_last_5 = sum(item["wickets"] for item in window)

        current_run_rate = score / overs_value

        # Use only the legal-ball positions available so far.
        window_overs = min(30, legal_balls) / 6.0
        last_five_run_rate = runs_last_5 / window_overs

        rows.append({
            "mid": str(match_id),
            "date": date,
            "venue": venue,
            "batting_team": TEAM_ALIASES.get(
                batting_team, batting_team
            ),
            "bowling_team": TEAM_ALIASES.get(
                bowling_team, bowling_team
            ),
            "batsman": delivery.get("batter", ""),
            "bowler": delivery.get("bowler", ""),
            "runs": score,
            "wickets": wickets,
            "overs": round(overs_value, 4),
            "runs_last_5": runs_last_5,
            "wickets_last_5": wickets_last_5,
            "current_run_rate": round(current_run_rate, 3),
            "last_five_run_rate": round(last_five_run_rate, 3),
            "striker": 0,
            "non-striker": 0,
            "total": final_total,
        })

    return rows


def convert_match(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        match = json.load(file)

    info = match.get("info", {})
    event = info.get("event") or {}
    competition = str(event.get("name", ""))

    if "indian premier league" not in competition.lower():
        return []

    dates = info.get("dates", [])
    date = str(dates[0]) if dates else ""
    venue = info.get("venue", "")
    teams = info.get("teams", [])

    rows = []

    for innings in match.get("innings", []):
        rows.extend(
            convert_innings(
                file_path.stem,
                date,
                venue,
                innings,
                teams,
            )
        )

    return rows


def main():
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Folder not found: {RAW_DIR}")

    files = sorted(RAW_DIR.rglob("*.json"))

    if not files:
        raise FileNotFoundError(f"No JSON files found in {RAW_DIR}")

    all_rows = []
    skipped = 0

    for index, file_path in enumerate(files, start=1):
        try:
            all_rows.extend(convert_match(file_path))
        except (
            json.JSONDecodeError,
            KeyError,
            TypeError,
            ValueError,
        ) as exc:
            skipped += 1
            print(f"Skipped {file_path.name}: {exc}")

        if index % 100 == 0:
            print(f"Processed {index}/{len(files)} files")

    if not all_rows:
        raise ValueError(
            "No IPL rows generated. Check the competition names."
        )

    df = pd.DataFrame(all_rows, columns=COLUMNS)
    df["total"] = pd.to_numeric(df["total"], errors="coerce")
    df = df.dropna(subset=["total"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nConversion complete")
    print(f"JSON files: {len(files)}")
    print(f"Files skipped: {skipped}")
    print(f"Rows: {len(df):,}")
    print(f"Matches: {df['mid'].nunique():,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Output: {OUTPUT_FILE}")
    print("Added features: current_run_rate, last_five_run_rate")


if __name__ == "__main__":
    main()
