# Credit Default Risk — Prediction Framework & Scorecard

**Role:** Junior Analyst, Retail Lending Risk Team  
**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 loan applicants  
**Objective:** Build a credit default prediction framework to support credit committee decisions on approval, decline, and manual review.

---

## Business Context

A retail lender needs to distinguish high-risk applicants from creditworthy ones — at scale, and with quantified cost of error. This project delivers:
- A three-model comparison (Logistic Regression, XGBoost, LightGBM)
- A cost-benefit threshold analysis in rupee terms
- A simplified credit scorecard for operational use
- A written recommendation memo to the Head of Credit Risk

---

## Project Structure

| Section | Business Question | Status |
|---------|------------------|--------|
| 1 — Data Quality Assessment | What does our applicant pool look like? | 🔄 In progress |
| 2 — Exploratory Data Analysis | Which borrower traits predict default? | ⬜ Pending |
| 3 — Feature Engineering | How do we translate data into risk signals? | ⬜ Pending |
| 4 — Model Building & Evaluation | Which model best separates defaulters? | ⬜ Pending |
| 5 — Cost-Benefit & Decision Framework | What does the right threshold cost? | ⬜ Pending |
| 6 — Scorecard & Recommendations | How do we operationalise this? | ⬜ Pending |

---

## Tech Stack

- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn
- **Models:** Logistic Regression · XGBoost · LightGBM
- **Metrics:** AUC-ROC · KS Statistic · Gini Coefficient · Cost-adjusted confusion matrix

---

## Key Results

*(To be updated as sections complete)*

---

## Files

| File | Description |
|------|-------------|
| `notebooks/credit_default_risk.ipynb` | Main analysis notebook |
| `outputs/scorecard.csv` | Three-tier scorecard output |
| `outputs/credit_committee_memo.pdf` | 2-page business recommendation |

---

## Data

Raw data not included (Kaggle Terms of Service). Download from:  
[kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit)  
Place `cs-training.csv` in the `/data` folder before running the notebook.

---

*Project built as part of a BFSI analytics portfolio. Business framing prioritised over model complexity.*
