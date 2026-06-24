# Cyberbullying Comment Classifier

This repository contains a Streamlit app for demonstrating a cyberbullying comment classification pipeline.

The app turns the notebook result into an interactive web interface. Users can type a comment, get a predicted label, inspect decision scores, and review the evaluation tables and charts.

## What the App Does

- Loads the prepared text-and-label dataset snapshot from `data/prepared_dataset.csv`.
- Trains the final tuned text classifier at startup.
- Uses raw word TF-IDF and character TF-IDF features.
- Applies the tuned decision offsets exported from the notebook.
- Predicts one of six labels:
  - `neutral_discussion`
  - `friendly_banter`
  - `harassment`
  - `malicious_sarcasm`
  - `trolling`
  - `prejudicial`
- Shows model comparison tables, classification report, confusion matrix, and tuning charts.

## Why Streamlit

The notebook is good for analysis, but a Streamlit app is easier to demonstrate. It lets someone test the model from a browser without running notebook cells manually.

## Project Files

```text
streamlit_app.py          Main Streamlit app
requirements.txt          Python dependencies for deployment
data/                     CSV files used by the app
.streamlit/config.toml    Streamlit theme/configuration
```

## Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Deploy to Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open Streamlit Community Cloud.
3. Create a new app.
4. Select the repository and branch.
5. Set the main file path to:

```text
streamlit_app.py
```

6. Deploy.

## Public or Private Repository

A public repository is simplest when the app link needs to be opened by anyone without signing in.

A private repository can also be used, but app access depends on sharing settings and invited viewers.

## Model Summary

The final model used in the app is:

```text
Optuna + Decision Offset Raw Word+Char Linear SVM
```

Evaluation from the notebook:

```text
Accuracy: 0.8689
Macro F1: 0.6302
```

Macro F1 is important because the dataset is imbalanced and contains minority classes.
