# Credit Default Risk — Prediction Framework & Scorecard

**Role:** Junior Analyst, Retail Lending Risk Team  
**Dataset:** [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit) — 150,000 U.S. loan applicants  
**Objective:** Build a credit default prediction framework to support credit committee decisions on approval, manual review, and decline.

---

## Business Context

A retail lender needs to distinguish high-risk applicants from creditworthy ones — at scale, and with a quantified cost of error. This project delivers:

- A three-model comparison (Logistic Regression, XGBoost, LightGBM)
- A cost-benefit threshold analysis expressed in rupee terms per 1,000 applicants
- A six-band credit scorecard (300–850 scale) for operational use
- A written recommendation memo to the Head of Credit Risk

The project prioritises **business thinking over model complexity**. Every feature, threshold, and metric is explained in terms a credit committee can act on.

---

## Key Results

### Model Performance

| Model | AUC-ROC | KS Statistic | Gini Coefficient |
|-------|--------:|-------------:|-----------------:|
| Logistic Regression | 0.8614 | 0.5683 | 0.7227 |
| XGBoost | 0.8692 | 0.5828 | 0.7384 |
| **LightGBM** | **0.8695** | **0.5865** | **0.7389** |

All three models exceed retail credit benchmarks (AUC > 0.75, KS > 0.30, Gini > 0.50).  
**Production model selected: LightGBM.**

### Cost-Benefit Analysis (per 1,000 applicants)

| Assumption | Value |
|------------|-------|
| Average loan size | ₹5,00,000 |
| Loss Given Default (LGD) | 60% |
| Net Interest Margin | 36% over loan life |
| Cost of missed defaulter (FN) | ₹3,00,000 |
| Cost of wrongly rejected borrower (FP) | ₹1,80,000 |
| FN:FP cost ratio | 1.7× |

**Optimal approval threshold: 0.59**  
At this threshold: 93.3% approval rate, 70.3% of defaulters caught.

### Credit Scorecard

| Score Band | Risk Tier | Recommended Action |
|------------|-----------|-------------------|
| 790–850 | Very Low Risk | Approve |
| 720–789 | Low Risk | Approve |
| 650–719 | Moderate Risk | Manual Review |
| 580–649 | Elevated Risk | Manual Review |
| 500–579 | High Risk | Decline |
| 300–499 | Very High Risk | Decline |

---

## Project Structure

| Section | Business Question |
|---------|------------------|
| 1 — Data Quality Assessment | What does our applicant pool look like? | 
| 2 — Exploratory Data Analysis | Which borrower traits predict default? | 
| 3 — Feature Engineering | How do we translate data into risk signals? | 
| 4 — Model Building & Evaluation | Which model best separates defaulters? | 
| 5 — Cost-Benefit & Decision Framework | What does the right threshold cost? | 
| 6 — Scorecard & Recommendations | How do we operationalise this? | 

---

## Section Summaries

### Section 1 — Data Quality Assessment

| Item | Finding |
|------|---------|
| Total applicants | 150,000 |
| Base default rate | 6.68% (14:1 class imbalance) |
| Rows removed | 1 (age = 0) |
| MonthlyIncome missing | 19.82% → imputed by age band median, missingness flag retained |
| NumberOfDependents missing | 2.62% → imputed by global median |
| RevolvingUtilization | Values up to 50,708 → capped at 1.0 |
| DebtRatio | Values up to 329,664 → capped at p99 (4,979) |
| MonthlyIncome | Values up to 3,008,750 → capped at p99 (25,000) |
| Sentinel codes | 96/98 in delinquency columns treated as missing, imputed to 0 |
| New feature | `income_missing` binary flag retained as model input |

### Section 3 — Feature Engineering

7 new features engineered from domain knowledge:

| Feature | Type | Top Correlation with Default |
|---------|------|------------------------------|
| `ever_seriously_delinquent` | Binary | +0.352 (strongest signal) |
| `ever_90day_delinquent` | Binary | +0.330 |
| `total_delinquencies` | Numeric | +0.229 |
| `high_utilisation` (>80%) | Binary | +0.259 |
| `young_borrower` (age < 35) | Binary | positive |
| `log_income`, `log_debt_ratio`, `log_utilisation` | Numeric | stabilises skew for LR |
| `lines_per_dependent` | Numeric | per-capita credit exposure |

Key sense checks:
- Borrowers with `ever_90day_delinquent = 1`: **41.23% default rate** vs 4.72% for clean borrowers
- Borrowers with `high_utilisation = 1`: **21.08% default rate** — 457% uplift over normal utilisation
- Borrowers aged under 35: **11.18% default rate** vs 6.02% for older borrowers

### Section 4 — Model Building & Evaluation

- **Train/test split:** 80/20, stratified to preserve 6.68% default rate in both sets
- **Imbalance handling:** `class_weight='balanced'` (LR), `scale_pos_weight=14` (XGBoost), `is_unbalance=True` (LightGBM)
- **Top features by XGBoost importance:** `total_delinquencies` (0.316), `ever_seriously_delinquent` (0.259), `high_utilisation` (0.077), `RevolvingUtilization` (0.075)

### Section 5 — Cost-Benefit & Decision Framework

Three-tier decision framework:

| Tier | Probability Band | % of Pool | Actual Default Rate |
|------|-----------------|-----------|---------------------|
| APPROVE | < 0.49 | 75.5% | 1.88% |
| MANUAL REVIEW | 0.49 – 0.55 | 4.3% | 7.54% |
| DECLINE | ≥ 0.55 | 20.2% | ~35%+ |

### Section 6 — Scorecard & Recommendations

Five operational recommendations made to the credit committee:
1. Adopt the three-tier decision framework for straight-through processing
2. Implement quarterly threshold recalibration
3. Retain `income_missing` flag in production — non-disclosure is a risk signal
4. Scope a reject inference exercise for the next model iteration
5. Schedule annual model rebuild

---

## Tech Stack

| Component | Tools |
|-----------|-------|
| Language | Python 3.13 |
| Data manipulation | Pandas, NumPy |
| Modelling | Scikit-learn, XGBoost, LightGBM |
| Visualisation | Matplotlib, Seaborn |
| Metrics | AUC-ROC, KS Statistic, Gini Coefficient |

---

## Repository Structure

credit-default-risk/
├── data/
│   ├── cs-training.csv          # Raw data (not included — Kaggle ToS)
│   ├── credit_clean.csv         # Post Section 1 cleaning
│   └── credit_engineered.csv    # Post Section 3 feature engineering
├── notebooks/
│   └── credit_default_risk.ipynb
├── outputs/
│   ├── s1_class_imbalance.png
│   ├── s1_missing_values.png
│   ├── s3_log_transform_income.png
│   ├── s3_feature_correlation.png
│   ├── s4_roc_comparison.png
│   ├── s4_feature_importance.png
│   ├── s5_cost_curve.png
│   ├── s5_threshold_tradeoffs.png
│   ├── s6_scorecard.png
│   ├── scorecard.csv
│   └── credit_committee_memo.txt
└── README.md


---

## Data

Raw data not included (Kaggle Terms of Service).  
Download `cs-training.csv` from [kaggle.com/c/GiveMeSomeCredit](https://www.kaggle.com/c/GiveMeSomeCredit) and place in `/data` before running the notebook.

---

## Limitations

- Dataset is U.S.-based. Indian credit behaviour, bureau coverage, and regulatory norms differ — thresholds should be validated on domestic portfolio data before production deployment
- No reject inference applied — model trained only on historically approved applicants
- Cost assumptions (LGD, NIM) are illustrative; production calibration requires lender-specific loss data
- Model does not account for macroeconomic regime changes

---

*Built as part of a BFSI analytics portfolio. Business framing prioritised over model complexity.*