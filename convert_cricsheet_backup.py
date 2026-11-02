
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
    "mid", "date", "venue", "batting_team", "bowling_team",
    "batsman", "bowler", "runs", "wickets", "overs",
    "runs_last_5", "wickets_last_5", "striker",
    "non-striker", "total",
]


def is_counted_wicket(wicket):
    return wicket.get("kind", "").lower() not in {
        "retired hurt",
        "obstructing the field",
    }


def convert_innings(match_id, date, venue, innings, teams):
    batting_team = innings.get("team", "")
    bowling_team = next(
        (team for team in teams if team != batting_team), ""
    )

    # A delivery event records its cumulative legal-ball number,
    # total runs and wickets on that delivery.
    history = []
    score = 0
    wickets = 0
    legal_balls = 0

    # Calculate the final innings total before creating training rows.
    for over_block in innings.get("overs", []):
        for delivery in over_block.get("deliveries", []):
            score += int(delivery.get("runs", {}).get("total", 0))
            wickets += sum(
                is_counted_wicket(w)
                for w in delivery.get("wickets", [])
            )

    final_total = score

    # Reset the counters and process the innings chronologically.
    score = 0
    wickets = 0
    legal_balls = 0
    rows = []

    for over_block in innings.get("overs", []):
        for delivery in over_block.get("deliveries", []):
            runs_info = delivery.get("runs", {})
            delivery_runs = int(runs_info.get("total", 0))
            batter_runs = int(runs_info.get("batter", 0))
            extras = delivery.get("extras", {})
            wicket_events = delivery.get("wickets", [])

            legal = not (
                "wides" in extras or "noballs" in extras
            )

            score += delivery_runs
            wickets += sum(
                is_counted_wicket(w) for w in wicket_events
            )

            if legal:
                legal_balls += 1

            # Keep all deliveries, including extras, associated with
            # their legal-ball position for rolling statistics.
            history.append({
                "ball": legal_balls,
                "runs": delivery_runs,
                "wickets": sum(
                    is_counted_wicket(w) for w in wicket_events
                ),
            })

            # Start predictions after six legal balls.
            if legal_balls < 6:
                continue

            window = [
                item for item in history
                if item["ball"] > legal_balls - 30
            ]

            completed_overs, balls_in_over = divmod(
                legal_balls, 6
            )
            overs_value = float(
                f"{completed_overs}.{balls_in_over}"
            )

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
                "overs": overs_value,
                "runs_last_5": sum(x["runs"] for x in window),
                "wickets_last_5": sum(
                    x["wickets"] for x in window
                ),
                # These two legacy fields are not reliable
                # cumulative batter scores in this data source.
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
                file_path.stem, date, venue, innings, teams
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
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
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


if __name__ == "__main__":
    main()
