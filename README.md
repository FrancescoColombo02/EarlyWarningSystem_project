# Early Warning System for Financial Market Stress

Anomaly detection on weekly financial market data to flag **risk-off** periods (weeks of high systemic risk).
Final project of **Group 19** for the **Fintech** course at **Politecnico di Milano** (A.Y. 2024/25).

## Overview

Financial markets alternate between *risk-on* phases, when investors buy risky assets, and *risk-off* phases, when they sell them and move to safe havens. Risk-off phases often come with crises.
The goal of this project is an **early warning system**: a model that reads a set of market indicators and flags anomalous weeks.

We compare a statistical baseline, supervised models, unsupervised models and deep learning models on the same labeled dataset, and then combine several of them into an ensemble.

## Data

- Weekly Bloomberg time series provided by the course: **1,111 weeks × 42 features**. The features are equity and bond indices, currencies, commodities, interest rates and economic surprise indices.
- The label `Y` marks anomalous (risk-off) weeks: **237 anomalies (21.3%)**.
- **The dataset is not included in this repository**, because it is Bloomberg data distributed for the course only.

**Preprocessing**
- To make the series stationary, we take log-differences for indices, prices and currencies, and first differences for interest rates, since rates can be negative.
- All features are standardized.

## Models

| Family | Model |
|---|---|
| Baseline | Multivariate Gaussian (MVG), with threshold ε tuned on the validation set |
| Supervised | Random Forest (with Optuna hyperparameter tuning), XGBoost, LSTM classifier |
| Unsupervised | COPOD |
| Semi-supervised deep learning | LSTM Autoencoder with active learning |
| Ensemble | Autoencoder + XGBoost + Local Outlier Factor (normalized scores, averaged) |

Results are also visualized with **PCA** and **3D UMAP** projections.

## Results

Test set: 207 weeks, of which 119 are anomalies. All models below are evaluated on this same split.

| Model | Precision | Recall | F1 |
|---|---|---|---|
| MVG (baseline) | 0.603 | 0.983 | 0.748 |
| Random Forest | 1.000 | 0.025 | 0.049 |
| Random Forest + Optuna | 0.944 | 0.286 | 0.439 |
| XGBoost (threshold 0.5) | 0.754 | 0.824 | 0.787 |
| COPOD | 0.669 | 0.748 | 0.706 |
| **LSTM Autoencoder + active learning*** | **0.825** | **0.992** | **0.901** |
| Ensemble AE + XGBoost + LOF (threshold 0.5) | 0.756 | 0.782 | 0.769 |

\* Evaluated on 20-week sequences built from the test set. The threshold is chosen on the validation set.

The **LSTM classifier** uses a separate chronological split with a 3-week window and class weights. Its test set has 222 weeks and 29 anomalies, and it reaches precision 0.696, recall 0.552 and F1 0.615.

**Takeaways**
- XGBoost is the strongest supervised model.
- Random Forest has very high precision but misses most anomalies, even after tuning.
- COPOD, which uses no labels, performs worse than both XGBoost and the MVG baseline.
- The LSTM Autoencoder with active learning gives the best F1 score.

## Limitations

- Anomalies are few and unevenly distributed over time, which makes training and evaluation harder.
- Most models use a shuffled train/validation/test split, which ignores the time order of the data. Only the LSTM classifier uses a chronological split.
- The notebook also reports results for XGBoost and the ensemble with a threshold chosen through ROC (Youden's J) on the test set itself. Those numbers are optimistic, so they are not included in the table above.

## Repository structure

```
EarlyWarningSystem_FinalProject.ipynb   # full analysis: preprocessing, models, evaluation, plots
README.md
```

The exploratory data analysis was done in R and the written report was prepared separately. Neither is included here.

## How to run

1. Open the notebook in **Google Colab**.
2. Upload `Dataset4_EWS.xlsx` to your Google Drive and update the file path in the notebook.
3. Run all cells. The notebook installs any packages Colab is missing.

Main libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`, `optuna`, `pyod`, `umap-learn`, `tensorflow`, `torch`.

## Team

Group 19, Politecnico di Milano:
- Francesco Colombo
- [Name Surname]
- [Name Surname]

**My contribution:** [describe your part, e.g. "XGBoost and ensemble models, hyperparameter tuning with Optuna"]

## Acknowledgements

The starting notebook and the dataset were provided by the course instructors.
