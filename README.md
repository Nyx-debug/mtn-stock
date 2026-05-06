# MTN Stock Prediction Dashboard

Simple school project for MTN Nigeria stock price prediction using Python and Streamlit.

## Project Overview

This project loads MTN stock data from `data/mtn_data.csv`, trains a Random Forest regression model, evaluates its performance, and displays results in a Streamlit dashboard.

## Files

- `app.py` - Streamlit dashboard app showing the latest market snapshot, model predictions, and price chart.
- `main.py` - Runs the full pipeline: data loading, preprocessing, training, prediction, and evaluation.
- `model.py` - Contains functions to train, save, and load the model.
- `preprocess.py` - Prepares features from raw stock data for the model.
- `data_collector.py` - Loads and cleans the CSV dataset.
- `evaluate.py` - Calculates model metrics and saves evaluation results.
- `data/mtn_data.csv` - Raw stock price data used for training and charting.
- `models/` - Stores the trained model file.
- `results/` - Stores evaluation output such as `metrics.json`.

## Requirements

- Python 3.14 or compatible
- pandas
- numpy
- scikit-learn
- joblib
- streamlit

Optional:
- matplotlib (only required if you want the evaluation script to save a plot image)

## Installation

Use a Python virtual environment and install the packages:

```bash
python -m venv venv
venv\Scripts\activate
pip install pandas numpy scikit-learn joblib streamlit
```

If you want evaluation plots too:

```bash
pip install matplotlib
```

## How to Run

1. Train the model and evaluate it:

```bash
python main.py
```

2. Open the dashboard:

```bash
streamlit run app.py
```

Then open the browser URL shown by Streamlit (usually `http://localhost:8501`).

## Notes

- The app uses the latest row from `data/mtn_data.csv` for snapshot values.
- The chart shows closing prices and a 10-period moving average.
- The current model is a simple Random Forest regressor suitable for a school project.

## What Works

- data loading and cleaning
- feature engineering for stock prediction
- training and saving a model
- displaying a Streamlit dashboard with live model predictions

## Project Goal

This project demonstrates a simple machine learning pipeline and interactive dashboard for stock price prediction using publicly available data and Python tools.

## Deployment

### Streamlit Community Cloud

1. Push your project to GitHub.
2. Go to https://streamlit.io/cloud and sign in with GitHub.
3. Create a new app and connect your repository.
4. Set the main file to `app.py`.
5. Deploy and use the generated public URL.

### GitHub setup steps

1. Initialize Git in the project folder if needed:

```bash
git init
git add .
git commit -m "Initial commit"
```

2. Create a GitHub repository and push:

```bash
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```

### Notes

- Keep `models/mtn_model.pkl` in the repo so the app can start without retraining.
- `data/mtn_data.csv` should also stay in the repo for the dashboard.
- `results/` is ignored by `.gitignore` because it contains generated output.
