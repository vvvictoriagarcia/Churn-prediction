<h2>About this project</h2>

<p>Customer churn is one of the most expensive problems in subscription-based businesses. Acquiring a new customer costs 5–7× more than retaining an existing one, so being able to predict who is about to leave — before they do — has a direct impact on revenue.</p>

<p>This project builds an end-to-end churn prediction pipeline on a synthetic telecom dataset of 7,000 customers, covering the full ML workflow: exploratory analysis, feature engineering, model training, cross-validation, and explainability.</p>

<h2>Dataset</h2>

<p>7,000 customers · 15 features · 46.6% churn rate. Features include tenure, monthly charges, contract type, internet service, payment method, tech support usage, and number of support calls. Churn probability was modeled with a logistic signal to produce realistic correlations between features and the target.</p>

<h2>What I built</h2>

<ul>
  <li><strong>EDA</strong> — churn rate broken down by every categorical feature to identify which segments churn the most</li>
  <li><strong>Distribution analysis</strong> — overlaid histograms comparing churned vs retained customers across numeric features</li>
  <li><strong>Preprocessing pipeline</strong> — StandardScaler for numeric features + OneHotEncoder for categoricals, wrapped in a ColumnTransformer to avoid data leakage</li>
  <li><strong>Three classifiers trained and compared</strong> — Logistic Regression, Random Forest, and Gradient Boosting, each evaluated with 5-fold stratified cross-validation</li>
  <li><strong>ROC curve comparison</strong> — all three models plotted together to visualize trade-offs</li>
  <li><strong>Feature importance</strong> — top 15 drivers of churn extracted from the Random Forest</li>
</ul>

<h2>Results</h2>

<table>
  <thead>
    <tr><th>Model</th><th>Test AUC</th><th>CV AUC (5-fold)</th></tr>
  </thead>
  <tbody>
    <tr><td>Logistic Regression</td><td>0.744</td><td>0.732</td></tr>
    <tr><td>Gradient Boosting</td><td>0.726</td><td>0.716</td></tr>
    <tr><td>Random Forest</td><td>0.706</td><td>0.705</td></tr>
  </tbody>
</table>

<p>Logistic Regression came out on top — a good reminder that simpler models often win when features are well-engineered. The small gap between Test AUC and CV AUC across all models indicates no significant overfitting.</p>

<h2>Key churn drivers</h2>

<ul>
  <li><strong>Short tenure</strong> — new customers churn at much higher rates</li>
  <li><strong>Month-to-month contract</strong> — no lock-in means higher exit risk</li>
  <li><strong>High monthly charges</strong> — price sensitivity is a real signal</li>
  <li><strong>Electronic check payment</strong> — correlated with lower engagement</li>
  <li><strong>Fiber optic internet</strong> — possibly linked to unmet expectations</li>
  <li><strong>Support calls</strong> — more calls = more friction = higher churn risk</li>
</ul>

<h2>Technical decisions</h2>

<ul>
  <li>Used <strong>Pipeline + ColumnTransformer</strong> to ensure preprocessing is always fitted on train data only — no leakage into validation folds</li>
  <li><strong>StratifiedKFold</strong> to preserve class balance across folds given the near 50/50 churn rate</li>
  <li>AUC chosen as the primary metric over accuracy — more informative when both false positives and false negatives carry business cost</li>
</ul>
