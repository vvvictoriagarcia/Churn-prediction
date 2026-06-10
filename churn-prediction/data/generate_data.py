"""
Generates a realistic synthetic telecom customer churn dataset.
Run once: python data/generate_data.py
"""
import pandas as pd
import numpy as np
import os

SEED = 42
np.random.seed(SEED)
N = 7000

# ── Features ──────────────────────────────────────────────────────────────────
tenure          = np.random.exponential(scale=30, size=N).clip(1, 72).astype(int)
monthly_charges = np.random.normal(65, 25, N).clip(20, 120).round(2)
total_charges   = (monthly_charges * tenure * np.random.uniform(0.85, 1.0, N)).round(2)
num_products    = np.random.choice([1,2,3,4], N, p=[0.30,0.35,0.25,0.10])
contract        = np.random.choice(["Month-to-month","One year","Two year"], N, p=[0.55,0.25,0.20])
payment_method  = np.random.choice(["Electronic check","Mailed check","Bank transfer","Credit card"], N, p=[0.35,0.22,0.23,0.20])
internet        = np.random.choice(["Fiber optic","DSL","No"], N, p=[0.44,0.34,0.22])
tech_support    = np.random.choice(["Yes","No"], N, p=[0.40,0.60])
online_security = np.random.choice(["Yes","No"], N, p=[0.38,0.62])
senior_citizen  = np.random.choice([0,1], N, p=[0.84,0.16])
partner         = np.random.choice(["Yes","No"], N, p=[0.48,0.52])
dependents      = np.random.choice(["Yes","No"], N, p=[0.30,0.70])
paperless       = np.random.choice(["Yes","No"], N, p=[0.59,0.41])
support_calls   = np.random.poisson(1.5, N).clip(0, 10)

# ── Churn probability (realistic signal) ──────────────────────────────────────
churn_score = (
    -0.04 * tenure
    + 0.012 * monthly_charges
    - 0.4  * (contract == "One year")
    - 0.9  * (contract == "Two year")
    + 0.5  * (internet == "Fiber optic")
    - 0.3  * (tech_support == "Yes")
    - 0.25 * (online_security == "Yes")
    + 0.3  * (payment_method == "Electronic check")
    + 0.2  * senior_citizen
    - 0.15 * (num_products >= 3)
    + 0.15 * support_calls
    + np.random.normal(0, 0.4, N)
)
churn_prob = 1 / (1 + np.exp(-churn_score))
churn = (np.random.uniform(size=N) < churn_prob).astype(int)

df = pd.DataFrame({
    "customer_id":      [f"CUST{i:05d}" for i in range(1, N+1)],
    "tenure":           tenure,
    "monthly_charges":  monthly_charges,
    "total_charges":    total_charges,
    "num_products":     num_products,
    "support_calls":    support_calls,
    "senior_citizen":   senior_citizen,
    "partner":          partner,
    "dependents":       dependents,
    "contract":         contract,
    "payment_method":   payment_method,
    "internet_service": internet,
    "tech_support":     tech_support,
    "online_security":  online_security,
    "paperless_billing":paperless,
    "churn":            churn,
})

out = os.path.join(os.path.dirname(__file__), "raw", "churn_data.csv")
df.to_csv(out, index=False)
print(f"Saved: {out}  |  shape: {df.shape}")
print(f"Churn rate: {churn.mean():.1%}")
