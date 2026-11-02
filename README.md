# 🏏 IPL Score Predictor – Advanced

An end-to-end Machine Learning project that predicts the final innings score of an Indian Premier League (IPL) cricket match using current match conditions and historical ball-by-ball data.

The application is built with Python and Streamlit and provides an interactive dashboard for predicting scores, comparing machine learning models, and visualizing projected innings scores.

## 📌 Project Overview

Predicting the final score of a cricket innings is challenging because it depends on several factors, including the current score, wickets lost, overs completed, recent scoring rate, batting team, bowling team, and venue.

This project uses historical IPL data to train machine learning models that estimate the final innings score based on the match situation.

**Best-performing model:** Extra Trees Regressor

## ✨ Features

- Predicts the estimated final IPL innings score.
- Interactive dashboard built using Streamlit.
- Dark-themed cricket-inspired user interface.
- Inputs for batting team, bowling team, venue, score, overs, and wickets.
- Considers runs scored and wickets lost in the last five overs.
- Calculates current run rate and recent run rate.
- Displays projected final score and additional expected runs.
- Visualizes the current score and projected score at 20 overs.
- Compares multiple machine learning models.
- Displays model evaluation metrics, including MAE, RMSE, and R².
- Uses a chronological train-test split to evaluate performance on later matches.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical calculations |
| Scikit-learn | Machine learning and model evaluation |
| Joblib | Saving and loading trained models |
| Streamlit | Interactive web dashboard |
| Plotly | Interactive score projection graph |
| JSON | Processing raw Cricsheet match data |
| Git and GitHub | Version control and project hosting |

## 📂 Project Structure

```text
IPL-Score-Predictor-Advanced/
│
├── data/
│   ├── cricsheet_raw/
│   │   └── Raw IPL match JSON files
│   │
│   └── processed/
│       └── cricsheet_ipl.csv
│
├── models/
│   ├── model_bundle.joblib
│   └── metadata.json
│
├── app.py
├── ipl_score_pred.py
├── train_models.py
├── requirements.txt
├── README.md
└── .gitignore
```

*Note: The structure above describes the main project files. Your local project may contain additional files and folders.*

## 📊 Dataset

The project uses IPL match data from Cricsheet, including ball-by-ball match information stored in JSON files.

The processed dataset contains approximately:

- **282,442** ball-by-ball records.
- **1,243** matches.
- **60** venues.
- **15** team categories.

The dataset covers matches from April 2008 to May 2026 in the current processed version.

### Data preprocessing

The raw JSON match files are processed into a structured CSV dataset. Match information is transformed into features that can be used by machine learning models.

The prediction system uses these ten features:

1. `batting_team`
2. `bowling_team`
3. `venue`
4. `runs`
5. `wickets`
6. `overs`
7. `runs_last_5`
8. `wickets_last_5`
9. `current_run_rate`
10. `last_five_run_rate`

The target variable is `total`, representing the final innings score.

## 🤖 Machine Learning Models

The project evaluates five machine learning model configurations:

- Random Forest
- Extra Trees
- Hist Gradient Boosting
- Hist Gradient Boosting – Deeper
- Hist Gradient Boosting – Faster

The model with the best recorded evaluation results is **Extra Trees**.

### Model evaluation results

The models were evaluated using a chronological split by match date, with the latest 20% of matches reserved for testing.

| Model | MAE (runs) | RMSE (runs) | R² |
|---|---:|---:|---:|
| Random Forest | 22.025 | 29.610 | 0.2931 |
| Extra Trees | 21.779 | 29.211 | 0.3121 |
| Hist Gradient Boosting | 22.223 | 29.418 | 0.3023 |
| Hist Gradient Boosting – Deeper | 22.181 | 29.345 | 0.3058 |
| Hist Gradient Boosting – Faster | 22.312 | 29.528 | 0.2971 |

### Understanding the results

- **MAE (Mean Absolute Error):** Extra Trees has an MAE of 21.779 runs, meaning its predictions differ from actual scores by approximately 21.8 runs on average on the evaluation set.
- **RMSE (Root Mean Squared Error):** The RMSE is 29.211 runs and gives greater weight to larger prediction errors.
- **R² Score:** The R² value is 0.3121, indicating limited explanatory performance compared with a perfect score of 1.

These results are useful for comparing the tested models, but they also show that score prediction can be improved further.

## 🚀 Installation and Setup

### Prerequisites

Install the following before running the project:

- Python 3.10 or a compatible version supported by the project dependencies.
- Git (optional, for cloning the repository).
- Visual Studio Code or another Python editor.

### Step 1: Clone the repository

```bash
git clone https://github.com/bhosaleshardul5-cmd/IPL-Score-Predictor-Advanced.git
cd IPL-Score-Predictor-Advanced
```

If your GitHub repository uses a different name, replace the repository URL and folder name with the correct ones.

### Step 2: Create a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
```

### Step 3: Activate the virtual environment

```powershell
.\venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Step 5: Run the application

From the project root directory:

```powershell
streamlit run app.py
```

The application should open in your browser. If it does not, visit:

```text
http://localhost:8501
```

### Using the existing trained model

The project includes saved model artifacts in the `models/` directory. Make sure `model_bundle.joblib` and `metadata.json` are present before launching the application.

You do not need to retrain the models just to run the dashboard when the saved artifacts and dependencies are available.

## 🎯 How to Use the Application

1. Start the Streamlit application.
2. Select the batting team and bowling team.
3. Select the match venue.
4. Enter the current innings score.
5. Specify the overs completed and balls in the current over.
6. Enter the number of wickets lost.
7. Enter runs and wickets from the last five overs.
8. Select the machine learning model, if model selection is available.
9. Click the prediction button.
10. Review the projected final score, additional expected runs, run rates, and score projection graph.

The prediction is an estimate based on the supplied match conditions and the model's learned patterns.

## 📈 Future Improvements

Possible future improvements include:

- Adding more recent and historical match data.
- Improving model accuracy through feature engineering and hyperparameter tuning.
- Including innings number, toss information, and match format where appropriate.
- Evaluating performance separately by venue, innings, and match situation.
- Adding prediction intervals to communicate uncertainty.
- Comparing predictions with simple baseline models.
- Improving the dashboard with additional match analytics.
- Deploying the application online so users can access it without running it locally.

## ⚠️ Limitations

- Predictions depend on the quality and relevance of the historical data.
- Actual match outcomes can differ considerably from predicted scores.
- The current evaluation metrics indicate that prediction accuracy can be improved.
- The model's evaluation uses matches after the training period, but future IPL conditions may differ from historical patterns.
- This project is intended for educational and analytical purposes; predictions are not guaranteed to match actual results.

## 👨‍💻 Author

**Shardul Sanjay Bhosale**

Third-Year CSE (AI & ML) Engineering Student

- GitHub: [bhosaleshardul5-cmd](https://github.com/bhosaleshardul5-cmd)
- LinkedIn: [Shardul Bhosale](https://linkedin.com/in/shardul-bhosale-21s24z30)

## 📜 Acknowledgements

Thanks to [Cricsheet](https://cricsheet.org/) for providing cricket match data that supports cricket analytics and machine learning projects.

---

⭐ If you find this project interesting, consider giving the repository a star on GitHub.
