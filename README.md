# Loan Default Risk Predictor

A machine learning project that predicts whether a loan applicant is likely to default, built as part of my capstone project. It walks through the full pipeline — from messy raw applicant data to a live, interactive prediction tool anyone can try.

**Try it live:** https://loan-default-predictor-fuvwsjx3h7xhsfziaxgjcf.streamlit.app/

## The Problem

Lenders need to know, before approving a loan, how likely an applicant is to default. Get it wrong in one direction and you turn away good customers; get it wrong in the other and you take on bad debt. This project tackles that trade-off using real applicant data (income, credit history, employment, and credit bureau scores) to estimate default risk.

## The Data

Built on the Home Credit Default Risk dataset (`application_train.csv`) — tens of thousands of loan applications, each labeled with whether the applicant ultimately defaulted. Like most real-world credit data, it's heavily imbalanced: the vast majority of applicants repay their loans, and only a small fraction default. That imbalance shaped almost every modeling decision below.

## What I Did

- Cleaned and explored the data, handling missing values and skewed distributions (e.g. log-transforming income)
- Engineered new features, including credit-to-income and annuity-to-income ratios, and frequency-encoded high-cardinality categorical fields like organization and occupation type
- Trained and compared three models: **Logistic Regression**, **Decision Tree**, and **Random Forest**, each tuned via `GridSearchCV`
- Evaluated models on more than just accuracy — since the dataset is imbalanced, accuracy alone is misleading. I focused on **recall** for the default class (how many actual defaulters the model catches), since missing a real defaulter is costlier to a lender than a false alarm

## Results

| Model | Accuracy | Precision (default) | Recall (default) | F1 (default) |
|---|---|---|---|---|
| Logistic Regression (tuned) | 68.5% | 0.16 | 0.68 | 0.26 |
| Decision Tree (tuned) | 59.8% | 0.13 | 0.72 | 0.22 |
| Random Forest (tuned) | 72.7% | 0.17 | 0.62 | 0.27 |

Random Forest edged out the others on overall accuracy and F1-score. But for the live demo, I chose the **tuned Logistic Regression model** — it catches a strong share of actual defaulters (68% recall) while staying more transparent and interpretable than a forest of trees, which matters in a lending context where decisions often need to be explainable.

## The Live Demo

Rather than leave the model buried in a notebook, I built a Streamlit web app so anyone — faculty, recruiters, or just the curious — can enter applicant details and get a real-time risk prediction, powered by the actual trained model.

**What it does:**
1. Takes applicant inputs (income, credit amount, employment history, credit bureau scores, etc.)
2. Reconstructs the same engineered features used in training
3. Scales the input exactly as the training data was scaled
4. Returns a live default probability and a clear risk classification

## Tech Stack

- **Python** — pandas, NumPy, scikit-learn for data processing and modeling
- **Streamlit** — for the live, interactive front end
- **joblib** — for saving and loading the trained model and preprocessing artifacts

## Files in This Repo

- `app.py` — the Streamlit application
- `loan_model.pkl`, `scaler.pkl`, `model_columns.pkl`, `column_medians.pkl`, `organization_freq_map.pkl`, `occupation_freq_map.pkl` — the trained model and preprocessing artifacts the app depends on
- `requirements.txt` — dependencies needed to run or deploy the app

## Running It Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
