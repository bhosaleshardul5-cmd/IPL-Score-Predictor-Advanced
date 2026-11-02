"""Professional Streamlit dashboard for IPL score prediction."""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from ipl_score_pred import load_models, predict_score


# ==================================================
# 1. CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
META_PATH = MODEL_DIR / "metadata.json"
BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"
DATA_PATH = ROOT / "data" / "processed" / "cricsheet_ipl.csv"

st.set_page_config(
    page_title="IPL Score Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# 2. CUSTOM UI STYLING
# ==================================================

st.markdown(
    """
    <style>
    /* Main application */
    .stApp {
        background-color: #0b1020;
        color: #f1f5f9;
    }

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.4px;
    }

    p, label, .stCaption {
        color: #cbd5e1;
    }

    /* Hero section */
    .hero {
        background: linear-gradient(125deg, #152746, #123b3a);
        border: 1px solid #285653;
        border-radius: 20px;
        padding: 30px;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 2.4rem;
        margin-bottom: 8px;
        color: #ffffff !important;
    }

    .hero p {
        color: #cbd5e1;
        font-size: 1rem;
        margin-bottom: 0;
    }

    .hero-tag {
        display: inline-block;
        background: #164e43;
        color: #86efac;
        border: 1px solid #28775e;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 12px;
    }

    /* Section containers */
    .section-card {
        background: #121a2b;
        border: 1px solid #263449;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 15px;
    }

    /* Input fields */
    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        background-color: #182236;
        border-color: #35445d;
        border-radius: 10px;
    }

    div[data-baseweb="select"] *,
    div[data-baseweb="input"] input {
        color: #f8fafc;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(90deg, #16a085, #0d9488);
        color: white;
        border: none;
        border-radius: 10px;
        min-height: 48px;
        font-weight: 700;
        transition: 0.2s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(90deg, #13856f, #0f766e);
        color: white;
        border: none;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #121a2b;
        border: 1px solid #263449;
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        color: #aab8cc;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    /* Alerts */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Data tables */
    div[data-testid="stDataFrame"] {
        border: 1px solid #263449;
        border-radius: 12px;
        overflow: hidden;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #0f1728;
        border-right: 1px solid #263449;
    }

    /* Dividers */
    hr {
        border-color: #263449;
    }

    /* Footer */
    .footer {
        text-align: center;
        color: #8492a8;
        font-size: 0.85rem;
        padding: 25px 0 5px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 3. HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-tag">DATA SCIENCE PROJECT</div>
        <h1>🏏 IPL Final Score Predictor</h1>
        <p>
            Predict the final innings score using machine learning,
            match statistics, and historical IPL data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# 4. LOAD MODELS
# ==================================================

if not META_PATH.exists() or not BUNDLE_PATH.exists():
    st.error(
        "Trained models were not found. Check the models folder "
        "for model_bundle.joblib and metadata.json."
    )
    st.stop()

try:
    bundle, metadata = load_models()
except Exception as exc:
    st.error(f"Could not load trained models: {exc}")
    st.stop()

if isinstance(bundle, dict):
    if isinstance(bundle.get("models"), dict):
        available_models = list(bundle["models"].keys())
    else:
        available_models = [
            name
            for name, model in bundle.items()
            if hasattr(model, "predict")
        ]
else:
    available_models = []

if not available_models:
    st.error("No trained models were found in the model bundle.")
    st.stop()

model_options = ["Best model"] + [
    name for name in available_models if name != "Best model"
]

best_model = metadata.get("best_model", "Unknown")


# ==================================================
# 5. TEAM AND VENUE OPTIONS
# ==================================================

team_options = [
    "Chennai Super Kings",
    "Delhi Capitals",
    "Gujarat Titans",
    "Kolkata Knight Riders",
    "Lucknow Super Giants",
    "Mumbai Indians",
    "Punjab Kings",
    "Rajasthan Royals",
    "Royal Challengers Bengaluru",
    "Sunrisers Hyderabad",
]

fallback_venues = [
    "M Chinnaswamy Stadium",
    "Wankhede Stadium",
    "Eden Gardens",
    "MA Chidambaram Stadium",
    "Arun Jaitley Stadium",
    "Narendra Modi Stadium",
    "Rajiv Gandhi International Stadium",
    "Sawai Mansingh Stadium",
    "Punjab Cricket Association Stadium",
    "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium",
]

venue_options = []

if DATA_PATH.exists():
    try:
        venue_data = pd.read_csv(
            DATA_PATH,
            usecols=["venue"],
            low_memory=False,
        )

        venue_options = sorted(
            venue_data["venue"]
            .dropna()
            .astype(str)
            .str.strip()
            .loc[lambda series: series != ""]
            .unique()
            .tolist()
        )
    except Exception:
        venue_options = []

if not venue_options:
    venue_options = fallback_venues


# ==================================================
# 6. SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("## 🏏 IPL Predictor")
    st.caption("Machine Learning Dashboard")

    st.divider()

    st.markdown("### About this project")
    st.write(
        "This application estimates an IPL innings' final score "
        "using historical match data and trained regression models."
    )

    st.markdown("### Best available model")
    st.info(str(best_model))

    st.markdown("### Model features")
    st.write(
        "- Batting and bowling teams\n"
        "- Match venue\n"
        "- Current runs and wickets\n"
        "- Overs completed\n"
        "- Recent scoring statistics"
    )

    st.divider()
    st.caption("Built with Python, Pandas, Plotly and Streamlit.")


# ==================================================
# 7. MATCH INPUT FORM
# ==================================================

st.subheader("🎯 Match Situation")
st.caption("Enter the current innings details to estimate the final score.")

with st.container(border=True):
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### 🏏 Batting details")

        batting_team = st.selectbox(
            "Batting team",
            team_options,
            index=5,
        )

        current_score = st.number_input(
            "Current score (runs)",
            min_value=0,
            max_value=299,
            value=100,
            step=1,
        )

        completed_overs = st.number_input(
            "Completed overs",
            min_value=0,
            max_value=19,
            value=15,
            step=1,
            help="Number of fully completed overs.",
        )

        balls_in_over = st.selectbox(
            "Legal balls in the current over",
            options=[0, 1, 2, 3, 4, 5],
            index=0,
            help="For example, 15 completed overs and 3 balls.",
        )

        wickets = st.number_input(
            "Wickets lost",
            min_value=0,
            max_value=10,
            value=3,
            step=1,
        )

    with right:
        st.markdown("#### 🎳 Bowling and match details")

        bowling_options = [
            team for team in team_options if team != batting_team
        ]

        bowling_team = st.selectbox(
            "Bowling team",
            bowling_options,
            index=0,
        )

        venue = st.selectbox(
            "Match venue",
            venue_options,
            index=0,
        )

        last_five_runs = st.number_input(
            "Runs in the recent five-over window",
            min_value=0,
            max_value=299,
            value=50,
            step=1,
            help=(
                "Runs in the last five overs, or the available "
                "innings if fewer than five overs have been bowled."
            ),
        )

        last_five_wickets = st.number_input(
            "Wickets lost in the recent five-over window",
            min_value=0,
            max_value=10,
            value=1,
            step=1,
        )

        model_name = st.selectbox(
            "Prediction model",
            model_options,
            index=0,
        )

    overs = completed_overs + balls_in_over / 6.0

    st.divider()

    info_left, info_right = st.columns(2)

    info_left.metric(
        "Overs bowled",
        f"{completed_overs}.{balls_in_over}",
    )

    info_right.metric(
        "Balls remaining",
        max(0, 120 - (completed_overs * 6 + balls_in_over)),
    )

    st.caption(
        "Cricket notation is used above: 15.3 means 15 overs "
        "and 3 balls, not 15.3 decimal overs."
    )

    predict_button = st.button(
        "🏆 Predict Final Score",
        type="primary",
        use_container_width=True,
    )


# ==================================================
# 8. MAKE PREDICTION
# ==================================================

if predict_button:
    if overs >= 20:
        st.error(
            "The innings has reached 20 overs. "
            "There are no remaining overs to predict."
        )

    elif current_score > 0 and overs == 0:
        st.error(
            "The score is greater than zero, but no balls "
            "have been bowled. Please check the inputs."
        )

    elif last_five_runs > current_score:
        st.error(
            "Runs in the recent five-over window cannot exceed "
            "the current score."
        )

    elif last_five_wickets > wickets:
        st.error(
            "Wickets lost in the recent five-over window cannot "
            "exceed the total wickets lost."
        )

    else:
        try:
            predicted = predict_score(
                batting_team=batting_team,
                bowling_team=bowling_team,
                runs=int(current_score),
                wickets=int(wickets),
                overs=overs,
                runs_last_5=int(last_five_runs),
                wickets_last_5=int(last_five_wickets),
                venue=venue,
                model_name=model_name,
            )

            actual_model = (
                best_model if model_name == "Best model" else model_name
            )

            additional_runs = max(
                0,
                int(predicted) - int(current_score),
            )

            # Prediction result
            st.divider()
            st.subheader("📊 Prediction Results")

            st.success(
                f"Estimated final score: {predicted} runs"
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Current score",
                f"{current_score}",
            )

            metric2.metric(
                "Predicted final score",
                f"{predicted}",
                help="Machine learning estimate for 20 overs.",
            )

            metric3.metric(
                "Projected additional runs",
                f"{additional_runs}",
            )

            st.caption(
                f"Model used: {actual_model} · "
                "Predictions are estimates, not guaranteed outcomes."
            )

            # Match statistics
            st.markdown("#### Match statistics")

            current_run_rate = (
                current_score / overs if overs > 0 else 0.0
            )

            recent_overs = min(5.0, overs)

            recent_run_rate = (
                last_five_runs / recent_overs
                if recent_overs > 0
                else 0.0
            )

            stat1, stat2, stat3 = st.columns(3)

            stat1.metric(
                "Current run rate",
                f"{current_run_rate:.2f}",
            )

            stat2.metric(
                "Recent run rate",
                f"{recent_run_rate:.2f}",
            )

            stat3.metric(
                "Wickets remaining",
                f"{10 - wickets}",
            )

            # --------------------------------------------------
            # SCORE PROJECTION GRAPH
            # --------------------------------------------------

            st.divider()
            st.subheader("📈 Score Projection")

            st.caption(
                "Compare the current score with the predicted "
                "score at the end of the innings."
            )

            projection_df = pd.DataFrame(
                {
                    "Overs": [overs, 20.0],
                    "Runs": [
                        int(current_score),
                        int(predicted),
                    ],
                    "Stage": [
                        "Current score",
                        "Predicted final score",
                    ],
                }
            )

            projection_chart = px.line(
                projection_df,
                x="Overs",
                y="Runs",
                markers=True,
                text="Runs",
                title="Current Score vs Predicted Final Score",
                labels={
                    "Overs": "Overs bowled",
                    "Runs": "Total runs",
                },
                hover_data=["Stage"],
            )

            projection_chart.update_traces(
                line={"width": 4, "color": "#14b8a6"},
                marker={
                    "size": 12,
                    "color": "#34d399",
                },
                textposition="top center",
            )

            projection_chart.update_layout(
                template="plotly_dark",
                paper_bgcolor="#121a2b",
                plot_bgcolor="#121a2b",
                font={"color": "#e2e8f0"},
                xaxis={
                    "title": "Overs",
                    "range": [0, 20],
                    "dtick": 2,
                    "gridcolor": "#263449",
                },
                yaxis={
                    "title": "Runs",
                    "rangemode": "tozero",
                    "gridcolor": "#263449",
                },
                height=420,
                margin={
                    "l": 20,
                    "r": 20,
                    "t": 60,
                    "b": 20,
                },
                hovermode="x unified",
            )

            st.plotly_chart(
                projection_chart,
                use_container_width=True,
            )

            st.caption(
                "This graph connects two points. It does not "
                "represent a ball-by-ball scoring forecast."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")


# ==================================================
# 9. MODEL EVALUATION
# ==================================================

st.divider()
st.subheader("🧪 Model Evaluation")

scores = metadata.get("scores", {})

if scores:
    score_df = pd.DataFrame(scores).T
    score_df.index.name = "Model"
    score_df = score_df.reset_index()

    with st.container(border=True):
        st.markdown("#### Model performance on test data")

        st.dataframe(
            score_df,
            use_container_width=True,
            hide_index=True,
        )

    if "MAE_runs" in score_df.columns:
        st.markdown("#### 📉 MAE Comparison")

        mae_chart = px.bar(
            score_df,
            x="Model",
            y="MAE_runs",
            title="Mean Absolute Error by Model",
            labels={
                "Model": "Machine learning model",
                "MAE_runs": "MAE (runs)",
            },
            text_auto=".2f",
        )

        mae_chart.update_layout(
            template="plotly_dark",
            paper_bgcolor="#121a2b",
            plot_bgcolor="#121a2b",
            font={"color": "#e2e8f0"},
            xaxis={
                "title": "Machine learning model",
                "tickangle": -20,
                "gridcolor": "#263449",
            },
            yaxis={
                "title": "Mean absolute error (runs)",
                "gridcolor": "#263449",
            },
            height=450,
            margin={
                "l": 20,
                "r": 20,
                "t": 60,
                "b": 80,
            },
        )

        st.plotly_chart(
            mae_chart,
            use_container_width=True,
        )

        st.caption(
            "Lower MAE indicates a smaller average absolute prediction error."
        )

    st.caption(
        "MAE and RMSE are measured in runs. R² is unitless. "
        "The metrics are calculated on a chronological held-out test split."
    )

else:
    st.info(
        "Model evaluation scores were not found in metadata."
    )


# ==================================================
# 10. PROJECT INFORMATION
# ==================================================

st.divider()
st.subheader("ℹ️ Project Information")

rows_used = metadata.get("rows_used", "unknown")

if isinstance(rows_used, (int, float)):
    rows_display = f"{rows_used:,}"
else:
    rows_display = str(rows_used)

dataset_name = metadata.get("dataset", "unknown")

info1, info2, info3 = st.columns(3)

info1.metric(
    "Training rows",
    rows_display,
)

info2.metric(
    "Available models",
    len(available_models),
)

info3.metric(
    "Best model",
    str(best_model),
)

with st.expander("About this project"):
    st.write(
        "The IPL Final Score Predictor uses historical IPL data "
        "to estimate the final first-innings score from the current "
        "match situation."
    )

    st.write(f"**Dataset:** `{dataset_name}`")

    st.write(
        "**Input features:** batting team, bowling team, venue, "
        "current runs, wickets, overs, recent runs and wickets, "
        "current run rate, and recent run rate."
    )

    st.warning(
        "Model performance can vary with team combinations, "
        "venues, match conditions and scoring patterns. "
        "Predictions should be treated as estimates."
    )


# ==================================================
# 11. FOOTER
# ==================================================

st.markdown(
    """
    <div class="footer">
        IPL Final Score Predictor · Machine Learning Project<br>
        Built with Python, Pandas, Plotly and Streamlit
    </div>
    """,
    unsafe_allow_html=True,
)