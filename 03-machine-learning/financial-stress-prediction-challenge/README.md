# Financial Stress Prediction

Predicting which mobile money customers are likely to experience **liquidity stress in the next 30 days** (`liquidity_stress_next_30d`), based on 6 months of transactional and behavioural history.

## Business context

Mobile money is often the primary financial tool available to customers, and early signs of liquidity stress in their transaction behaviour could allow providers to offer timely support before a customer's situation worsens. This project explores whether such signals exist in the data, and how strong they are, before moving on to a predictive model.

## The data

Each customer has 6 months of monthly history (`m1` = most recent, `m6` = oldest) across several feature groups:

| Group | Description |
|---|---|
| **Customer profile** | Demographics and account-level attributes (age, gender, ARPU, region, segment...) |
| **Balance** | Daily average balance per month |
| **Activity** | General account activity indicators |
| **Behaviour** | Monthly transaction metrics (volume, total value, highest amount, and a group-specific count) across 7 action categories: `merchantpay`, `paybill`, `deposit`, `withdraw`, `mm_send`, `received`, `transfer_from_bank` |
| **Target** | `liquidity_stress_next_30d` — binary, imbalanced (~15% positive) |

## Project structure

The project is split into two parts, each with its own notebook:

```
financial-stress-prediction-challenge/
├── 01_EDA_Financial_Stress_V5.ipynb                  # Part 1 — Exploratory Data Analysis (this repo's focus so far)
├── 02_Modeling_Financial_Stress.ipynb                # Part 2 — Feature Engineering, Modeling, Evaluation
├── utils.py                  # Reusable data-transformation functions (shared by both notebooks)
├── plot.py                   # Visualization functions used throughout the EDA
├── structure.md                  # Planned pipeline / section-by-section outline
├── data/                         # (gitignored — raw dataset, not tracked)
├── versions/                     # (gitignored — notebook drafts / iterations)
└── submissions/                  # (gitignored — challenge submission files)
```

> Adjust the notebook file names above to match your actual repo if they differ.

## Part 1 — EDA highlights

The EDA follows a hypothesis-driven structure: business understanding → data understanding → 6 initial hypotheses → data quality → univariate analysis → behavioural evolution over the 6-month window → bivariate/interaction analysis → formal hypothesis evaluation.

A recurring theme throughout is **separating participation from intensity**: several behavioural features are zero-inflated (a large share of customers simply don't use a given service every month), so raw averages can be misleading unless split into "does the customer use this service?" and "how much, among those who do?".

### Hypothesis evaluation summary

| Hypothesis | Verdict | Key evidence |
|---|---|---|
| H1: ↓ spending on goods/services (merchantpay) | **Reject** | Stressed customers spend *more*, not less, throughout the window |
| H2: ↓ bill payments (paybill) | **Reject** | Same direction as H1 |
| H3: ↓ cash inflows (deposit) | **Weak** | Driven by a participation drop (customers stop depositing), not by reduced deposits among active customers |
| H4: ↑ cash withdrawals (withdraw) | **Support** | Persistently elevated from the start of the window, reinforced by a later rise in participation |
| H5: ↑ external support (received, transfer_from_bank) | **Support** | Fewer instances of receiving support as stress approaches, but each instance grows in size |
| H6: Behavioural deterioration over time | **Weak** | Balance behaves inconsistently relative to its own baseline within the stressed group, but the signal is narrow |

**Overall pattern:** financial stress in this dataset doesn't look like broad, gradual belt-tightening — it looks like a small subset of customers who already withdraw more than average, who then partially disengage from deposits and external transfers a few months before the stress window, while spending on everyday goods and bills unchanged or higher. That combination points to a **liquidity squeeze**, consistent with the target's own definition, rather than a spending-driven cause.

## Tech stack

- Python (pandas, numpy, scipy)
- seaborn / matplotlib for visualization
- xgboost + shap for exploratory (non-final) interaction discovery between behavioural features

## Running the notebooks

```bash
pip install -r requirements.txt   # pandas, numpy, scipy, seaborn, matplotlib, xgboost, shap
```

Both notebooks import shared helper functions:

```python
from eda_utils import *   # data transformations (redundancy checks, monthly-feature collapsing, etc.)
from eda_plot import *    # plotting functions
```

## Next steps (Part 2)

- Feature engineering building on the EDA findings (participation flags, trend/slope features, redundancy-aware feature selection)
- Customer behaviour clustering
- Baseline and tuned models, with an evaluation focused on recall/precision trade-offs given the class imbalance and the cost asymmetry of missing a stressed customer
