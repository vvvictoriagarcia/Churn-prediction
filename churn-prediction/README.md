# Churn Prediction — Telecom Customer Retention Model

End-to-end machine learning project to predict customer churn in a telecom company. Covers data generation, exploratory analysis, feature engineering, model training, and evaluation of three classifiers.

---

## Results

| Model | Test AUC | CV AUC (5-fold) |
|---|---|---|
| Logistic Regression | 0.744 | 0.732 |
| Gradient Boosting | 0.726 | 0.716 |
| Random Forest | 0.706 | 0.705 |

**Best model:** Logistic Regression — AUC 0.744

**Top churn drivers:** tenure, monthly charges, contract type (month-to-month), electronic check payment, fiber optic internet, support calls.

---

## Project Structure

```
churn-prediction/
├── data/
│   ├── generate_data.py     # synthetic dataset generator
│   └── raw/                 # generated CSV (gitignored)
├── outputs/figures/         # exported charts (gitignored)
├── run_analysis.py          # full pipeline: EDA + training + evaluation
├── requirements.txt
└── .gitignore
```

---

## Analysis & Modeling Steps

1. **EDA** — churn rate by contract, internet service, payment method, tech support, security, seniority
2. **Distributions** — tenure, monthly charges, total charges, support calls split by churn label
3. **Correlation matrix** — numeric feature relationships
4. **Preprocessing** — StandardScaler for numerics, OneHotEncoder for categoricals via ColumnTransformer
5. **Training** — Logistic Regression, Random Forest, Gradient Boosting with 5-fold stratified CV
6. **Evaluation** — ROC curves, confusion matrix, classification report
7. **Explainability** — Random Forest feature importances (top 15)
8. **Model comparison** — Test AUC vs CV AUC side by side

---

## Setup

```bash
pip install -r requirements.txt
python data/generate_data.py
python run_analysis.py
```

---

## Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![matplotlib](https://img.shields.io/badge/matplotlib-11557C?style=flat-square)
![seaborn](https://img.shields.io/badge/seaborn-4C72B0?style=flat-square)
