import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_data(filepath="F:\\Datasets\\creditcard.csv"):
    df = pd.read_csv(filepath)
    print(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Basic info
    fraud_count = df['Class'].sum()
    fraud_pct = df['Class'].mean() * 100

    print(f"\n CLASS DISTRIBUTION:")
    print(f"  Legitimate: {len(df) - fraud_count:,} ({100 - fraud_pct:.3f}%)")
    print(f"  Fraud: {fraud_count:,} ({fraud_pct:.3f}%)")
    print(f" Severe class imbalance detected!")

    return df


def data_quality_check(df):
    print("DATA QUALITY CHECK : ")


    missing = df.isnull().sum().sum()
    print(f"  Missing values: {missing}")

    duplicates = df.duplicated().sum()
    print(f"  Duplicate rows: {duplicates}")

    print(f"\n  Data types:")
    print(df.dtypes.value_counts())

    if missing == 0 and duplicates == 0:
        print(f"\n  DATA QUALITY: EXCELLENT")
    else:
        print(f"\n DATA QUALITY: NEEDS ATTENTION")

    return missing, duplicates


def statistical_analysis(df):

    print("STATISTICAL HYPOTHESIS TESTING : ")

    print("\nMann-Whitney U Test (Non-parametric)")
    print("H0: No difference between fraud and legitimate distributions")
    print("H1: Distributions are significantly different")

    features = [col for col in df.columns if col not in ['Time', 'Class']]
    significant_features = []

    print(f"\nTesting {len(features)} features...\n")

    for feature in features:
        fraud_data = df[df['Class'] == 1][feature]
        legit_data = df[df['Class'] == 0][feature]

        statistic, p_value = stats.mannwhitneyu(fraud_data, legit_data,
                                                alternative='two-sided')

        if p_value < 0.01:
            significant_features.append((feature, p_value))

    significant_features.sort(key=lambda x: x[1])

    print(f" Significant features (p < 0.01): {len(significant_features)}/{len(features)}\n")
    print("Top 10 most significant features:")
    for i, (feature, p_val) in enumerate(significant_features[:10], 1):
        print(f"  {i:2d}. {feature:8s} → p-value: {p_val:.2e}")

    return significant_features


def correlation_analysis(df):
    print("CORRELATION ANALYSIS : ")
    correlations = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)

    print("\nTop 10 features most correlated with Fraud:\n")
    for i, (feature, corr) in enumerate(correlations.head(10).items(), 1):
        print(f"  {i:2d}. {feature:8s} → Correlation: {corr:.4f}")

    return correlations


def plot_class_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df['Class'].value_counts()

    axes[0].bar(['Legitimate', 'Fraud'], counts.values,
                color=['#2ecc71', '#e74c3c'], alpha=0.8, edgecolor='black', linewidth=2)
    axes[0].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[0].set_title('Class Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)

    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 5000, f'{v:,}', ha='center', fontweight='bold', fontsize=11)

    colors = ['#2ecc71', '#e74c3c']
    explode = (0, 0.1)
    axes[1].pie(counts.values, labels=['Legitimate', 'Fraud'], autopct='%1.4f%%',
                colors=colors, explode=explode, startangle=90, textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Class Imbalance', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
    print(" Class distribution plot displayed")


def plot_time_analysis(df):
    print("\n  Generating time analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.3, wspace=0.25)

    df_temp = df.copy()
    df_temp['Hour'] = (df_temp['Time'] / 3600) % 24
    df_temp['TimeHours'] = df_temp['Time'] / 3600

    # 1. Transaction timeline
    axes[0, 0].scatter(df_temp[df_temp['Class'] == 0]['TimeHours'],
                       df_temp[df_temp['Class'] == 0].index,
                       alpha=0.15, s=0.3, c='#3498db', label='Legitimate', rasterized=True)
    axes[0, 0].scatter(df_temp[df_temp['Class'] == 1]['TimeHours'],
                       df_temp[df_temp['Class'] == 1].index,
                       alpha=0.95, s=25, c='#e74c3c', label='Fraud', marker='D',
                       edgecolors='darkred', linewidths=0.5)
    axes[0, 0].set_xlabel('Time (hours)', fontweight='bold', fontsize=11)
    axes[0, 0].set_ylabel('Transaction Index', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('Transaction Timeline', fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].legend(loc='upper left', framealpha=0.9, fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')

    # 2. Fraud count by hour
    fraud_by_hour = df_temp[df_temp['Class'] == 1].groupby('Hour').size()
    axes[0, 1].bar(fraud_by_hour.index, fraud_by_hour.values,
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.5)
    axes[0, 1].set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
    axes[0, 1].set_ylabel('Fraud Count', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('Fraud Transactions by Hour', fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].set_xticks(range(0, 24, 2))
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')
    axes[0, 1].set_ylim(0, max(fraud_by_hour.values) * 1.15)

    # 3. Total transactions
    total_by_hour = df_temp.groupby('Hour').size()
    axes[1, 0].plot(total_by_hour.index, total_by_hour.values,
                    marker='o', linewidth=3, markersize=8, color='#3498db',
                    markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#3498db')
    axes[1, 0].fill_between(total_by_hour.index, total_by_hour.values, alpha=0.25, color='#3498db')
    axes[1, 0].set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
    axes[1, 0].set_ylabel('Total Transactions', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('Total Transactions by Hour', fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].set_xticks(range(0, 24, 2))
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].set_ylim(0, max(total_by_hour.values) * 1.1)

    # 4. Fraud rate
    fraud_rate = (fraud_by_hour / total_by_hour * 100).fillna(0)
    axes[1, 1].plot(fraud_rate.index, fraud_rate.values,
                    marker='s', linewidth=3, markersize=8, color='#e74c3c',
                    markerfacecolor='white', markeredgewidth=2.5, markeredgecolor='#e74c3c')
    axes[1, 1].fill_between(fraud_rate.index, fraud_rate.values, alpha=0.25, color='#e74c3c')
    axes[1, 1].set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
    axes[1, 1].set_ylabel('Fraud Rate (%)', fontweight='bold', fontsize=11)
    axes[1, 1].set_title('Fraud Rate by Hour', fontsize=14, fontweight='bold', pad=15)
    axes[1, 1].set_xticks(range(0, 24, 2))
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].set_ylim(0, max(fraud_rate.values) * 1.15)

    plt.show()
    print(" Time analysis plots displayed")


def plot_amount_analysis(df):
    print("\n  Generating amount analysis plots...")

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.subplots_adjust(hspace=0.25, wspace=0.3)

    legit_amounts = df[df['Class'] == 0]['Amount']
    fraud_amounts = df[df['Class'] == 1]['Amount']

    # 1. Histogram - Better colors
    axes[0, 0].hist(legit_amounts, bins=50, alpha=0.7,
                    label='Legitimate', color='#3498db', edgecolor='black', linewidth=0.5)
    axes[0, 0].hist(fraud_amounts, bins=50, alpha=0.7,
                    label='Fraud', color='#e74c3c', edgecolor='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Amount ($)', fontweight='bold', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontweight='bold', fontsize=11)
    axes[0, 0].set_title('Amount Distribution (0-500 range)', fontsize=14, fontweight='bold', pad=15)
    axes[0, 0].legend(loc='upper right', framealpha=0.9, fontsize=10)
    axes[0, 0].set_xlim(0, 500)
    axes[0, 0].grid(axis='y', alpha=0.3, linestyle='--')

    # 2. Log scale
    axes[0, 1].hist(legit_amounts[legit_amounts > 0], bins=50, alpha=0.7,
                    label='Legitimate', color='#3498db', log=True, edgecolor='black', linewidth=0.5)
    axes[0, 1].hist(fraud_amounts[fraud_amounts > 0], bins=50, alpha=0.7,
                    label='Fraud', color='#e74c3c', log=True, edgecolor='black', linewidth=0.5)
    axes[0, 1].set_xlabel('Amount ($)', fontweight='bold', fontsize=11)
    axes[0, 1].set_ylabel('Frequency (log scale)', fontweight='bold', fontsize=11)
    axes[0, 1].set_title('Amount Distribution (Full Range - Log)', fontsize=14, fontweight='bold', pad=15)
    axes[0, 1].legend(loc='upper right', framealpha=0.9, fontsize=10)
    axes[0, 1].grid(axis='y', alpha=0.3, linestyle='--')

    # 3. Box plot
    bp = axes[1, 0].boxplot([legit_amounts, fraud_amounts],
                            labels=['Legitimate', 'Fraud'],
                            patch_artist=True,
                            showmeans=True,
                            meanprops=dict(marker='D', markerfacecolor='yellow', markersize=8),
                            medianprops=dict(color='black', linewidth=2),
                            boxprops=dict(linewidth=1.5),
                            whiskerprops=dict(linewidth=1.5),
                            capprops=dict(linewidth=1.5))

    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#e74c3c')
    bp['boxes'][1].set_alpha(0.7)

    axes[1, 0].set_ylabel('Amount ($)', fontweight='bold', fontsize=11)
    axes[1, 0].set_title('Amount Distribution by Class', fontsize=14, fontweight='bold', pad=15)
    axes[1, 0].grid(axis='y', alpha=0.3, linestyle='--')

    # 4. Statistics table
    axes[1, 1].axis('off')
    stats_text = f"""AMOUNT STATISTICS

LEGITIMATE TRANSACTIONS:
  • Mean:      ${legit_amounts.mean():>10,.2f}
  • Median:    ${legit_amounts.median():>10,.2f}
  • Std Dev:   ${legit_amounts.std():>10,.2f}
  • Max:       ${legit_amounts.max():>10,.2f}

FRAUD TRANSACTIONS:
  • Mean:      ${fraud_amounts.mean():>10,.2f}
  • Median:    ${fraud_amounts.median():>10,.2f}
  • Std Dev:   ${fraud_amounts.std():>10,.2f}
  • Max:       ${fraud_amounts.max():>10,.2f}
"""
    axes[1, 1].text(0.15, 0.5, stats_text, fontsize=11.5, verticalalignment='center',
                    family='monospace', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor='lightblue',
                              alpha=0.3, edgecolor='black', linewidth=2))

    plt.show()
    print("Amount analysis plots displayed")


def plot_feature_distributions(df, features):
    print("\n  Generating feature distribution plots...")

    n_features = len(features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(17, 5.5 * n_rows))
    fig.subplots_adjust(hspace=0.35, wspace=0.3)
    axes = axes.flatten()

    for idx, feature in enumerate(features):
        ax = axes[idx]

        legit_data = df[df['Class'] == 0][feature]
        fraud_data = df[df['Class'] == 1][feature]

        legit_data.plot.kde(ax=ax, label='Legitimate', color='#3498db', linewidth=3)
        fraud_data.plot.kde(ax=ax, label='Fraud', color='#e74c3c', linewidth=3, linestyle='--')

        ax.set_xlabel(feature, fontweight='bold', fontsize=10)
        ax.set_ylabel('Density', fontweight='bold', fontsize=10)
        ax.legend(loc='best', framealpha=0.9, fontsize=9)
        ax.set_title(f'{feature} Distribution', fontsize=13, fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3, linestyle='--')

        ax.set_facecolor('#f8f9fa')

    for idx in range(n_features, len(axes)):
        axes[idx].axis('off')

    plt.show()
    print("  Feature distribution plots displayed")

def run_eda():
    print("\n")
    print("FRAUD DETECTION SYSTEM")
    print("EXPLORATORY DATA ANALYSIS : ")

    # 1. Load data
    df = load_data()
    data_quality_check(df)
    significant_features = statistical_analysis(df)
    correlations = correlation_analysis(df)

    # 5. Visualizations
    print("GENERATING VISUALIZATIONS : ")

    print("\nClass distribution")
    plot_class_distribution(df)

    print("\nTime patterns")
    plot_time_analysis(df)

    print("\nAmount analysis")
    plot_amount_analysis(df)

    print("\nFeature distributions")
    top_features = [f for f, _ in significant_features[:9]]
    plot_feature_distributions(df, top_features)

    print(" KEY FINDINGS & INSIGHTS : ")

    fraud_pct = df['Class'].mean() * 100
    print(f"""
    1. CLASS IMBALANCE:
       • Fraud rate: {fraud_pct:.3f}%
       • Imbalance ratio: 1:{int(1 / df['Class'].mean())}
       → CRITICAL: Must use specialized techniques (SMOTE, class weights)

    2. DISCRIMINATIVE FEATURES:
       • {len(significant_features)} out of {len(df.columns) - 2} features are significant
       • Top features: {', '.join([f for f, _ in significant_features[:5]])}
       → Strong predictive signals available

    3. TEMPORAL PATTERNS:
       • Fraud varies by time of day
       • Need to create time-based features

    4. AMOUNT PATTERNS:
       • Different distributions for fraud vs legitimate
       • Amount is a valuable feature
    """)

    return df

# PHASE 2: FEATURE ENGINEERING & DATA PREPARATION

def engineer_features(df):
    print("FEATURE ENGINEERING : ")


    df_new = df.copy()

    # 1. Time-based features
    print("\n Creating time-based features...")
    df_new['Hour'] = (df_new['Time'] / 3600) % 24
    df_new['Day'] = (df_new['Time'] / 86400).astype(int)

    def get_time_category(hour):
        if 0 <= hour < 6:
            return 0
        elif 6 <= hour < 12:
            return 1
        elif 12 <= hour < 18:
            return 2
        else:
            return 3

    df_new['TimeCategory'] = df_new['Hour'].apply(get_time_category)

    # 2. Amount-based features
    print(" Creating amount-based features...")
    df_new['Amount_log'] = np.log1p(df_new['Amount'])
    df_new['Amount_squared'] = df_new['Amount'] ** 2

    def categorize_amount(amount):
        if amount <= 10:
            return 0
        elif amount <= 50:
            return 1
        elif amount <= 100:
            return 2
        elif amount <= 500:
            return 3
        else:
            return 4

    df_new['Amount_category'] = df_new['Amount'].apply(categorize_amount)

    # 3. Statistical features from PCA components
    print("Creating statistical features...")
    v_features = [col for col in df_new.columns if col.startswith('V')]

    # Mean and std of V features
    df_new['V_mean'] = df_new[v_features].mean(axis=1)
    df_new['V_std'] = df_new[v_features].std(axis=1)
    df_new['V_min'] = df_new[v_features].min(axis=1)
    df_new['V_max'] = df_new[v_features].max(axis=1)
    df_new['V_range'] = df_new['V_max'] - df_new['V_min']

    # 4. Interaction features
    print("Creating interaction features...")
    # Based on correlation analysis, V14, V4, V12 are important
    df_new['V14_V4'] = df_new['V14'] * df_new['V4']
    df_new['V14_Amount'] = df_new['V14'] * df_new['Amount_log']
    df_new['V12_V10'] = df_new['V12'] * df_new['V10']

    # 5. Frequency encoding
    print("Creating frequency-based features...")
    # Hour frequency
    hour_freq = df_new.groupby('Hour').size() / len(df_new)
    df_new['Hour_frequency'] = df_new['Hour'].map(hour_freq)

    # Day frequency
    day_freq = df_new.groupby('Day').size() / len(df_new)
    df_new['Day_frequency'] = df_new['Day'].map(day_freq)

    print(f"\n Feature engineering complete!")
    print(f"  Original features: {len(df.columns)}")
    print(f"  New features created: {len(df_new.columns) - len(df.columns)}")
    print(f"  Total features: {len(df_new.columns)}")

    # Show new features
    new_features = [col for col in df_new.columns if col not in df.columns]
    print(f"\n  New features: {', '.join(new_features)}")

    return df_new
def prepare_data_for_modeling(df):
    print("DATA PREPARATION FOR MODELING : ")
    df_sorted = df.sort_values('Time').reset_index(drop=True)
    print("   Reason: In production, we predict FUTURE frauds using PAST data")

    # Define splits: 60% train, 20% validation, 20% test
    n = len(df_sorted)
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)

    # Split data
    train_df = df_sorted[:train_size].copy()
    val_df = df_sorted[train_size:train_size + val_size].copy()
    test_df = df_sorted[train_size + val_size:].copy()

    print(f"\n Dataset splits:")
    print(f"  Training:   {len(train_df):>6,} ({len(train_df) / n * 100:.1f}%) - "
          f"Fraud: {train_df['Class'].sum():>3,} ({train_df['Class'].mean() * 100:.3f}%)")
    print(f"  Validation: {len(val_df):>6,} ({len(val_df) / n * 100:.1f}%) - "
          f"Fraud: {val_df['Class'].sum():>3,} ({val_df['Class'].mean() * 100:.3f}%)")
    print(f"  Test:       {len(test_df):>6,} ({len(test_df) / n * 100:.1f}%) - "
          f"Fraud: {test_df['Class'].sum():>3,} ({test_df['Class'].mean() * 100:.3f}%)")


    cols_to_drop = ['Class', 'Time']

    X_train = train_df.drop(columns=cols_to_drop)
    y_train = train_df['Class']

    X_val = val_df.drop(columns=cols_to_drop)
    y_val = val_df['Class']

    X_test = test_df.drop(columns=cols_to_drop)
    y_test = test_df['Class']

    print(f"\n Feature matrix shape:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")

    print("\n Applying feature scaling (StandardScaler)...")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

    print(" Scaling complete!")

    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler


def visualize_feature_importance_before_modeling(df):
    print("FEATURE IMPORTANCE PREVIEW : ")

    correlations = df.corr()['Class'].drop('Class').abs().sort_values(ascending=False)

    # Top 20 features
    fig, ax = plt.subplots(figsize=(12, 8))

    top_20 = correlations.head(20)
    colors = ['#e74c3c' if corr > 0.1 else '#3498db' for corr in top_20.values]

    bars = ax.barh(range(len(top_20)), top_20.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(top_20)))
    ax.set_yticklabels(top_20.index, fontweight='bold')
    ax.set_xlabel('Absolute Correlation with Fraud', fontsize=12, fontweight='bold')
    ax.set_title('Top 20 Features by Correlation with Fraud', fontsize=14, fontweight='bold', pad=20)
    ax.grid(axis='x', alpha=0.3, linestyle='--')


    for i, (bar, val) in enumerate(zip(bars, top_20.values)):
        ax.text(val, bar.get_y() + bar.get_height() / 2, f'{val:.4f}',
                va='center', ha='left', fontsize=9, fontweight='bold', color='black')

    plt.tight_layout()
    plt.show()

    print(f"\n Top 5 most predictive features:")
    for i, (feat, corr) in enumerate(top_20.head(5).items(), 1):
        print(f"  {i}. {feat:20s} → Correlation: {corr:.4f}")


def run_phase2(df):
    print("PHASE 2: FEATURE ENGINEERING & DATA PREPARATION")

    df_engineered = engineer_features(df)
    visualize_feature_importance_before_modeling(df_engineered)

    X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data_for_modeling(df_engineered)

    return df_engineered, X_train, X_val, X_test, y_train, y_val, y_test, scaler


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve, auc,
                             roc_curve, precision_score, recall_score, f1_score)


def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall_vals, precision_vals)

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    return {
        'Model':     model_name,
        'Accuracy':  accuracy,
        'Precision': precision,
        'Recall':    recall,
        'F1-Score':  f1,
        'ROC-AUC':   roc_auc,
        'PR-AUC':    pr_auc,
        'FPR':       fpr,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }


def print_evaluation_report(results):
    print(f"MODEL: {results['Model']}")
    print(f"\n  Accuracy:  {results['Accuracy']:.4f}")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f}")
    print(f"  F1-Score:  {results['F1-Score']:.4f}")
    print(f"  ROC-AUC:   {results['ROC-AUC']:.4f}")
    print(f"  PR-AUC:    {results['PR-AUC']:.4f}  ← Key metric")
    print(f"\n  TP: {results['TP']:,}  FP: {results['FP']:,}  "
          f"TN: {results['TN']:,}  FN: {results['FN']:,}")
    fraud_total = results['TP'] + results['FN']
    detection_rate = results['TP'] / fraud_total if fraud_total > 0 else 0
    print(f"  Fraud Detection Rate: {detection_rate*100:.2f}%  |  "
          f"FP Rate: {results['FPR']*100:.2f}%")


def train_naive_baseline(y_train, y_val):
    print("BASELINE 1: NAIVE (Always Predict Not Fraud)")


    y_pred = np.zeros(len(y_val))
    y_pred_proba = np.zeros(len(y_val))

    results = evaluate_model(y_val, y_pred, y_pred_proba, "Naive Baseline")
    print_evaluation_report(results)
    return results


def train_logistic_regression(X_train, y_train, X_val, y_val, class_weight=None):
    model_name = "Logistic Regression" if class_weight is None else "Logistic Regression (Weighted)"
    print(f"BASELINE: {model_name.upper()}")

    model = LogisticRegression(max_iter=1000, class_weight=class_weight,
                               random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    results = evaluate_model(y_val, y_pred, y_pred_proba, model_name)
    print_evaluation_report(results)
    return model, results


def train_decision_tree(X_train, y_train, X_val, y_val):
    print("BASELINE: DECISION TREE")


    model = DecisionTreeClassifier(max_depth=10, min_samples_split=100,
                                   min_samples_leaf=50, class_weight='balanced',
                                   random_state=42)
    model.fit(X_train, y_train)

    y_pred       = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    results = evaluate_model(y_val, y_pred, y_pred_proba, "Decision Tree")
    print_evaluation_report(results)
    return model, results


def train_random_forest(X_train, y_train, X_val, y_val):
    print("BASELINE: RANDOM FOREST")


    model = RandomForestClassifier(n_estimators=100, max_depth=15,
                                   min_samples_split=100, min_samples_leaf=50,
                                   class_weight='balanced', random_state=42,
                                   n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    results = evaluate_model(y_val, y_pred, y_pred_proba, "Random Forest")
    print_evaluation_report(results)
    return model, results


def compare_baseline_models(results_list):
    print("BASELINE MODEL COMPARISON")


    comp = pd.DataFrame(results_list)[
        ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'PR-AUC', 'FPR']
    ]

    print("\n" + comp.to_string(index=False))

    best_idx   = comp['PR-AUC'].idxmax()
    best_model = comp.loc[best_idx, 'Model']
    best_score = comp.loc[best_idx, 'PR-AUC']
    print(f"\n  Best Model: {best_model}  |  PR-AUC: {best_score:.4f}")

    return comp


def plot_confusion_matrices(results_list):

    n   = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results_list):
        cm = np.array([[res['TN'], res['FP']],
                       [res['FN'], res['TP']]])

        sns.heatmap(cm, annot=True, fmt=',', cmap='Blues', ax=ax,
                    cbar=False, linewidths=2, linecolor='black',
                    annot_kws={'fontsize': 12, 'fontweight': 'bold'})

        ax.set_title(res['Model'], fontsize=11, fontweight='bold', pad=12)
        ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
        ax.set_ylabel('Actual',    fontsize=10, fontweight='bold')
        ax.set_xticklabels(['Legit', 'Fraud'], fontweight='bold')
        ax.set_yticklabels(['Legit', 'Fraud'], fontweight='bold', rotation=0)

    plt.suptitle('Confusion Matrices — Baseline Models',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

def visualize_model_comparison(results_list):
    print("\n  Generating comparison visualizations...")
    df_viz = pd.DataFrame(results_list)

    model_colors = {
        'Naive Baseline':                '#95a5a6',
        'Logistic Regression':           '#3498db',
        'Logistic Regression (Weighted)':'#2980b9',
        'Decision Tree':                 '#e67e22',
        'Random Forest':                 '#27ae60'
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.subplots_adjust(hspace=0.35, wspace=0.35)

    #1. PR-AUC  (top-left)
    ax  = axes[0, 0]
    sdf = df_viz.sort_values('PR-AUC', ascending=True)
    col = [model_colors.get(m, '#34495e') for m in sdf['Model']]
    bars = ax.barh(range(len(sdf)), sdf['PR-AUC'],
                   color=col, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(sdf)))
    ax.set_yticklabels(sdf['Model'], fontsize=10, fontweight='bold')
    ax.set_xlabel('PR-AUC Score', fontsize=11, fontweight='bold')
    ax.set_title('PR-AUC Comparison (Primary Metric)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, sdf['PR-AUC']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    #2. Recall  (top-right)
    ax  = axes[0, 1]
    sdf = df_viz.sort_values('Recall', ascending=True)
    col = [model_colors.get(m, '#34495e') for m in sdf['Model']]
    bars = ax.barh(range(len(sdf)), sdf['Recall'],
                   color=col, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(sdf)))
    ax.set_yticklabels(sdf['Model'], fontsize=10, fontweight='bold')
    ax.set_xlabel('Recall (Fraud Detection Rate)', fontsize=11, fontweight='bold')
    ax.set_title('Recall Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, sdf['Recall']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    #3. Precision vs Recall scatter  (bottom-left)
    ax = axes[1, 0]
    for _, row in df_viz.iterrows():
        c = model_colors.get(row['Model'], '#34495e')
        ax.scatter(row['Recall'], row['Precision'],
                   s=400, alpha=0.85, color=c, edgecolors='black', linewidths=2)
    ax.set_xlabel('Recall', fontsize=11, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax.set_title('Precision vs Recall Trade-off',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, linestyle='--')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=model_colors.get(m, '#34495e'),
                             edgecolor='black', label=m)
                       for m in df_viz['Model']],
              loc='lower right', fontsize=8, framealpha=0.95)

    #4. F1-Score  (bottom-right)
    ax  = axes[1, 1]
    sdf = df_viz.sort_values('F1-Score', ascending=True)
    col = [model_colors.get(m, '#34495e') for m in sdf['Model']]
    bars = ax.barh(range(len(sdf)), sdf['F1-Score'],
                   color=col, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax.set_yticks(range(len(sdf)))
    ax.set_yticklabels(sdf['Model'], fontsize=10, fontweight='bold')
    ax.set_xlabel('F1-Score', fontsize=11, fontweight='bold')
    ax.set_title('F1-Score Comparison', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    for bar, val in zip(bars, sdf['F1-Score']):
        ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.show()

def run_phase3(X_train, y_train, X_val, y_val):
    results_list = []
    models_dict  = {}

    results = train_naive_baseline(y_train, y_val)
    results_list.append(results)

    model, results = train_logistic_regression(X_train, y_train, X_val, y_val, class_weight=None)
    results_list.append(results)
    models_dict['Logistic Regression'] = model

    model, results = train_logistic_regression(X_train, y_train, X_val, y_val, class_weight='balanced')
    results_list.append(results)
    models_dict['Logistic Regression (Weighted)'] = model

    model, results = train_decision_tree(X_train, y_train, X_val, y_val)
    results_list.append(results)
    models_dict['Decision Tree'] = model

    model, results = train_random_forest(X_train, y_train, X_val, y_val)
    results_list.append(results)
    models_dict['Random Forest'] = model

    comparison_df = compare_baseline_models(results_list)


    visualize_model_comparison(results_list)
    plot_confusion_matrices(results_list)

    best_model_name = comparison_df.loc[comparison_df['PR-AUC'].idxmax(), 'Model']
    best_pr_auc = comparison_df['PR-AUC'].max()

    print(f"  Best model: {best_model_name}  |  PR-AUC: {best_pr_auc:.4f}")

    return models_dict, results_list, comparison_df
if __name__ == "__main__":
    print("STARTING FRAUD DETECTION PROJECT")

    df = run_eda()

    df_engineered, X_train, X_val, X_test, y_train, y_val, y_test, scaler = run_phase2(df)

    models_dict, results_list, comparison_df = run_phase3(X_train, y_train, X_val, y_val)
