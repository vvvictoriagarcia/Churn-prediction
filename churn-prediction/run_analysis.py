"""
Full churn prediction pipeline:
  - EDA
  - Preprocessing
  - Model training (Logistic Regression, Random Forest, Gradient Boosting)
  - Evaluation (ROC, Confusion Matrix, Feature Importance)
  - Saves all figures to outputs/figures/
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)
COLORS = sns.color_palette('muted')

BASE   = Path(__file__).parent
FIG_DIR = BASE / 'outputs' / 'figures'
FIG_DIR.mkdir(parents=True, exist_ok=True)

def save(name):
    plt.savefig(FIG_DIR / f'{name}.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  saved {name}.png')

# ── 1. Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(BASE / 'data' / 'raw' / 'churn_data.csv')
print(f'Dataset: {df.shape}  |  Churn rate: {df.churn.mean():.1%}')

# ── 2. EDA — churn rate by feature ───────────────────────────────────────────
cat_features = ['contract','internet_service','payment_method',
                'tech_support','online_security','senior_citizen']
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
axes = axes.flatten()
for ax, col in zip(axes, cat_features):
    rates = df.groupby(col)['churn'].mean().sort_values(ascending=False)
    bars = ax.bar(rates.index.astype(str), rates.values * 100,
                  color=COLORS[:len(rates)], alpha=0.85, edgecolor='white')
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.3, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=9)
    ax.set_title(col.replace('_',' ').title(), fontweight='bold')
    ax.set_ylabel('Churn Rate (%)')
    ax.set_ylim(0, rates.values.max()*100 * 1.2)
    ax.set_xticklabels(rates.index.astype(str), rotation=20, ha='right')
plt.suptitle('Churn Rate by Key Features', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
save('01_churn_rate_by_feature')

# ── 3. Numeric distributions by churn ────────────────────────────────────────
num_features = ['tenure', 'monthly_charges', 'total_charges', 'support_calls']
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, col in zip(axes, num_features):
    for val, label, color in [(0,'No Churn',COLORS[0]),(1,'Churned',COLORS[3])]:
        ax.hist(df[df.churn==val][col], bins=30, alpha=0.6, label=label, color=color, density=True)
    ax.set_title(col.replace('_',' ').title(), fontweight='bold')
    ax.set_xlabel('')
    if col == num_features[0]:
        ax.legend(fontsize=9)
plt.suptitle('Feature Distributions: Churned vs Retained', fontsize=13, fontweight='bold')
plt.tight_layout()
save('02_numeric_distributions')

# ── 4. Correlation heatmap ────────────────────────────────────────────────────
num_cols = ['tenure','monthly_charges','total_charges','num_products','support_calls','senior_citizen','churn']
corr = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, linewidths=0.5, ax=ax)
ax.set_title('Correlation Matrix', fontweight='bold')
plt.tight_layout()
save('03_correlation_heatmap')

# ── 5. Preprocessing ──────────────────────────────────────────────────────────
df = df.drop(columns=['customer_id'])
X = df.drop(columns=['churn'])
y = df['churn']

num_cols = ['tenure','monthly_charges','total_charges','num_products','support_calls','senior_citizen']
cat_cols = ['contract','payment_method','internet_service','tech_support',
            'online_security','partner','dependents','paperless_billing']

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print(f'Train: {X_train.shape}  |  Test: {X_test.shape}')

# ── 6. Train models ───────────────────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
}

results = {}
for name, model in models.items():
    pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=StratifiedKFold(5), scoring='roc_auc')
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, y_prob)
    results[name] = {'pipe': pipe, 'y_prob': y_prob, 'auc': auc, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std()}
    print(f'  {name:25s}  AUC={auc:.4f}  CV={cv_scores.mean():.4f} +/- {cv_scores.std():.4f}')

# ── 7. ROC curves ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for i, (name, r) in enumerate(results.items()):
    fpr, tpr, _ = roc_curve(y_test, r['y_prob'])
    ax.plot(fpr, tpr, lw=2, color=COLORS[i], label=f"{name} (AUC = {r['auc']:.3f})")
ax.plot([0,1],[0,1],'k--', lw=1, label='Random baseline')
ax.fill_between([0,1],[0,1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Model Comparison', fontweight='bold', fontsize=13)
ax.legend(fontsize=10); ax.grid(True, alpha=0.4)
plt.tight_layout()
save('04_roc_curves')

# ── 8. Confusion matrix (best model) ─────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['auc'])
best      = results[best_name]
y_pred    = best['pipe'].predict(X_test)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax,
    colorbar=False, cmap='Blues',
    display_labels=['Retained','Churned'])
ax.set_title(f'Confusion Matrix — {best_name}', fontweight='bold')
plt.tight_layout()
save('05_confusion_matrix')

print(f'\nBest model: {best_name}')
print(classification_report(y_test, y_pred, target_names=['Retained','Churned']))

# ── 9. Feature importance (Random Forest) ────────────────────────────────────
rf_pipe = results['Random Forest']['pipe']
ohe_cols = list(rf_pipe.named_steps['pre']
                .named_transformers_['cat']
                .get_feature_names_out(cat_cols))
all_features = num_cols + ohe_cols

importances = pd.Series(
    rf_pipe.named_steps['clf'].feature_importances_, index=all_features
).sort_values(ascending=True).tail(15)

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(importances.index, importances.values,
               color=COLORS[0], alpha=0.85, edgecolor='white')
ax.set_title('Top 15 Feature Importances — Random Forest', fontweight='bold', fontsize=13)
ax.set_xlabel('Importance')
plt.tight_layout()
save('06_feature_importance')

# ── 10. Model comparison bar chart ────────────────────────────────────────────
comp = pd.DataFrame([
    {'Model': k, 'Test AUC': v['auc'], 'CV AUC': v['cv_mean']}
    for k, v in results.items()
])
fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(len(comp))
w = 0.35
ax.bar(x - w/2, comp['Test AUC'], w, label='Test AUC', color=COLORS[0], alpha=0.85, edgecolor='white')
ax.bar(x + w/2, comp['CV AUC'],   w, label='CV AUC (5-fold)', color=COLORS[1], alpha=0.85, edgecolor='white')
ax.set_xticks(x); ax.set_xticklabels(comp['Model'])
ax.set_ylim(0.5, 1.0); ax.set_ylabel('AUC Score')
ax.set_title('Model Comparison — AUC Scores', fontweight='bold', fontsize=13)
ax.legend()
for i, row in comp.iterrows():
    ax.text(i - w/2, row['Test AUC'] + 0.003, f"{row['Test AUC']:.3f}", ha='center', fontsize=9)
    ax.text(i + w/2, row['CV AUC']   + 0.003, f"{row['CV AUC']:.3f}",   ha='center', fontsize=9)
plt.tight_layout()
save('07_model_comparison')

print('\nAll figures saved to outputs/figures/')
