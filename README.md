# 🔍 Fraud Detection System
> Master's Level Data Science & Statistical Learning Project

![Python](https://img.shields.io/badge/Python-3.12-blue)
![sklearn](https://img.shields.io/badge/scikit--learn-1.3-orange)
![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)

## Overview
An end-to-end **fraud detection system** built using advanced statistical 
learning techniques on highly imbalanced real-world financial transaction data.

**Key Challenges Addressed:**
- Extreme class imbalance (only 0.173% fraud rate — 1 in every 578 transactions)
- Feature engineering from anonymized PCA-transformed transaction data
- Rigorous statistical validation using Mann-Whitney U hypothesis testing
- Business-oriented evaluation using PR-AUC over standard accuracy


## Dataset
**Source:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

- 284,807 total transactions over 48 hours
- 492 fraud cases — only 0.173% of all transactions
- 30 features (V1–V28 are PCA-transformed, plus Time and Amount)
- Imbalance ratio of 1:578 (for every fraud, there are 578 legitimate transactions)
- 1,081 duplicate rows detected during quality check

## Project Structure
fraud_detection_system/
│
├── fraud_detection_project.py   ← Complete project (all phases)
├── graphs/                      ← All saved visualizations
│   ├── class_distribution.png
│   ├── time_pattern.png
│   ├── amount_analysis.png
│   ├── feature_distribution.png
│   ├── top20_features.png
│   ├── correlation_plots.png
│   └── comparison_charts.png
├── data/
│   ├── raw/                     ← Dataset (not tracked — too large)
│   └── processed/
├── requirements.txt
├── .gitignore
└── README.md

## 🔬 Phases Completed
### ✅ Phase 1: Exploratory Data Analysis

#### Data Quality
The dataset contained no missing values, which is ideal for modeling. However, 1,081 duplicate rows were detected and flagged for attention. The data consists of 30 float64 columns and 1 integer column (the target Class label).

#### Statistical Hypothesis Testing — Mann-Whitney U Test
A Mann-Whitney U test was applied to all 29 features to check whether their distributions differ significantly between fraud and legitimate transactions. The null hypothesis states no difference exists between the two groups. Out of 29 features tested, 25 were statistically significant at the 99% confidence level (p < 0.01). The top features by significance were V14, V4, V12, V11, and V10, with p-values as extreme as 1.47e-260, indicating overwhelmingly strong evidence of distributional differences between fraud and legitimate transactions.

#### Correlation with Fraud
Looking at absolute correlations with the fraud label, V17 came out on top at 0.3265, followed by V14 at 0.3025 and V12 at 0.2606. While these individual correlations may seem modest, they are remarkably strong given that the features are PCA-transformed and the fraud rate is only 0.173%.

#### Visualizations

**Class Distribution**
![Class Distribution](graphs/class_distribution.png)

**Temporal Patterns**
![Time Pattern](graphs/time_pattern.png)

**Amount Analysis**
![Amount Analysis](graphs/amount_analysis.png)

**Feature Distributions**
![Feature Distribution](graphs/feature_distribution.png)

### Phase 2: Feature Engineering & Data Preparation

#### New Features Created
Starting from 31 original features, 16 new features were engineered bringing the total to 47. These fell into five categories.

**Time-based** — The raw Time column was converted into Hour of day, Day number, and a TimeCategory label (Night, Morning, Afternoon, Evening) to capture behavioral patterns across the day.

**Amount-based** — The Amount column was log-transformed to reduce skewness, squared to capture nonlinear effects, and binned into five categories ranging from very small to very large transactions.

**Statistical** — Across all 28 PCA features (V1–V28), the mean, standard deviation, minimum, maximum, and range were computed per transaction, giving the model a summary view of each row's overall profile.

**Interaction features** — Three cross-feature multiplications were created: V14 × V4, V14 × log(Amount), and V12 × V10. These proved to be the most powerful features in the entire dataset.

**Frequency encoding** — The transaction frequency of each hour and each day was computed and added as a feature, giving the model awareness of how unusual a given time slot is.

#### Most Predictive Features After Engineering
The interaction features dramatically outperformed all original features. V14 × V4 achieved a correlation of 0.5749 with the fraud label, and V12 × V10 reached 0.5486 — nearly double the correlation of the best individual feature V17 (0.3265). This confirms that fraud behavior is captured not by single features alone but by combinations of them.

![Top 20 Features](graphs/top20_features.png)

#### Data Split Strategy
A temporal split was used rather than a random split. This is critical because in production, a model always predicts future transactions using patterns learned from past ones. Randomising the split would leak future information into training and produce unrealistically optimistic results.

The data was sorted by time and divided into 60% training (170,884 transactions, 360 fraud cases), 20% validation (56,961 transactions, 57 fraud cases), and 20% test (56,962 transactions, 75 fraud cases). All features were then scaled using StandardScaler fitted only on training data.

---

### Phase 3: Baseline Models

Five models were trained and evaluated on the validation set of 56,961 transactions containing 57 fraud cases.

#### Model Results

The **Naive Baseline** always predicts legitimate, achieving 99.90% accuracy while catching zero fraud. This highlights why accuracy is a misleading metric for imbalanced data.

**Logistic Regression** without class weighting detected 40 out of 57 frauds (70.18% recall) with only 6 false alarms, giving a strong precision of 0.87. However it misses nearly 30% of fraud cases which is unacceptable in a real system.

**Logistic Regression with class weighting** pushed recall up to 89.47%, detecting 51 out of 57 frauds. The trade-off is a much lower precision (0.058) and 828 false alarms. It achieved the best PR-AUC of 0.7896, making it the strongest baseline overall.

**Decision Tree** detected 47 out of 57 frauds but with 663 false alarms and the weakest PR-AUC of 0.6503 among the non-naive models, suggesting it overfits to the majority class pattern.

**Random Forest** struck a reasonable balance — detecting 44 out of 57 frauds with only 28 false alarms and a precision of 0.61. Its PR-AUC of 0.7558 and very low false positive rate of 0.05% make it the most production-friendly baseline despite not having the highest recall.

#### Summary
PR-AUC is used as the primary metric because standard accuracy is completely misleading when 99.83% of transactions are legitimate. The best baseline model is Logistic Regression (Weighted) with a PR-AUC of 0.7896, detecting 89.47% of all frauds. However, all five baselines are expected to be significantly surpassed in Phase 4 using gradient boosting and SMOTE.

![Comparison Charts](graphs/comparison_charts.png)

## 🛠️ Tech Stack

The project is built entirely in Python 3.12. Data processing uses pandas and numpy. Visualisations are produced with matplotlib and seaborn. Statistical testing uses scipy. Machine learning models come from scikit-learn. Gradient boosting models use xgboost and lightgbm. Class imbalance is handled with imbalanced-learn.

---
## ⚙️ How to Run

### 1. Clone the repository
```bash
git clone https://github.com/YourUsername/fraud-detection-system.git
cd fraud-detection-system
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
- Go to [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download `creditcard.csv`
- Place it inside `data/raw/`

### 4. Run
```bash
python fraud_detection_project.py
```

## Key Findings So Far

1. **Severe Imbalance** — 0.173% fraud, ratio 1:578 — accuracy is meaningless, PR-AUC is the metric
2. **Data Quality** — 1,081 duplicate rows detected and flagged
3. **Strong Signals** — 25/29 features statistically significant (p < 0.01)
4. **Interaction Features Win** — V14×V4 correlation 0.5749 vs V14 alone 0.3025
5. **Class Weighting** — Boosts fraud detection from 70% → 89% recall
6. **Precision-Recall Trade-off** — LR (Weighted) catches most fraud but generates 828 false alarms vs Random Forest's 28

## References
1. Dal Pozzolo et al. (2018) — *Learned lessons in credit card fraud detection*
2. He & Garcia (2009) — *Learning from Imbalanced Data*
3. Chen & Guestrin (2016) — *XGBoost: A Scalable Tree Boosting System*
4. Chawla et al. (2002) — *SMOTE: Synthetic Minority Over-sampling Technique*

