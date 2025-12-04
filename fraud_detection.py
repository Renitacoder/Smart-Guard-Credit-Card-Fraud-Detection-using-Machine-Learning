import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

np.random.seed(42)
n_samples = 5000

df = pd.DataFrame({
    "amount": np.random.exponential(scale=200, size=n_samples),
    "time": np.random.randint(0, 24, n_samples),
    "location_risk": np.random.randint(0, 2, n_samples),
    "device_trust": np.random.randint(0, 2, n_samples),
    "past_fraud": np.random.randint(0, 2, n_samples),
})
fraud_prob = (
    0.3 * (df["amount"] > 250).astype(int) +
    0.25 * df["location_risk"] +
    0.2 * (df["time"].isin([0,1,2,3,23])).astype(int) +
    0.25 * df["past_fraud"]
)

df["fraud"] = np.random.binomial(1, np.clip(fraud_prob, 0, 1))
print(df.head())
print(df["fraud"].value_counts())
X = df.drop("fraud", axis=1)
y = df["fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    print(f"\n{name} Results:")
    print(classification_report(y_test, y_pred))
    
    results[name] = {
        "confusion": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob)
    }
plt.figure(figsize=(5,4))
sns.heatmap(results["Random Forest"]["confusion"], annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
plt.figure(figsize=(6,5))
for name, model in models.items():
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, y_prob):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()
