"""
============================================================
  PRODIGY INFOTECH - DATA SCIENCE INTERNSHIP | TASK 03
  Decision Tree Classifier — Bank Marketing Dataset
  Tools: Python, Pandas, Scikit-learn, Matplotlib, Seaborn
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 1. CREATE BANK MARKETING-LIKE DATASET
# ─────────────────────────────────────────────
# If you have the real CSV, replace with:
#   df = pd.read_csv("bank.csv", sep=";")

np.random.seed(42)
n = 1000

age      = np.random.randint(18, 70, n)
job      = np.random.choice(["admin","technician","services",
                              "management","retired","student"], n)
marital  = np.random.choice(["married","single","divorced"], n,
                             p=[0.55, 0.32, 0.13])
education= np.random.choice(["primary","secondary","tertiary"], n,
                             p=[0.20, 0.50, 0.30])
balance  = np.random.randint(-500, 5000, n)
housing  = np.random.choice(["yes","no"], n, p=[0.55, 0.45])
loan     = np.random.choice(["yes","no"], n, p=[0.16, 0.84])
contact  = np.random.choice(["cellular","telephone","unknown"], n,
                             p=[0.65, 0.15, 0.20])
duration = np.random.randint(0, 600, n)
campaign = np.random.randint(1, 10, n)
poutcome = np.random.choice(["success","failure","unknown"], n,
                             p=[0.12, 0.38, 0.50])

# Purchase probability based on features
prob = (
    0.10
    + 0.20 * (poutcome == "success")
    + 0.15 * (duration > 300)
    + 0.05 * (education == "tertiary")
    + 0.05 * (balance > 1000)
    - 0.05 * (loan == "yes")
    - 0.05 * (campaign > 5)
    + np.random.normal(0, 0.05, n)
).clip(0, 1)

y_label = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "age": age, "job": job, "marital": marital,
    "education": education, "balance": balance,
    "housing": housing, "loan": loan,
    "contact": contact, "duration": duration,
    "campaign": campaign, "poutcome": poutcome,
    "purchased": y_label
})

print("=" * 55)
print("  PRODIGY INFOTECH | TASK 03 — Decision Tree Classifier")
print("=" * 55)
print(f"\n📋 Dataset Shape : {df.shape}")
print(f"   Purchase Rate : {df['purchased'].mean()*100:.1f}%")
print(f"   No Purchase   : {(1-df['purchased'].mean())*100:.1f}%")


# ─────────────────────────────────────────────
# 2. DATA PREPROCESSING
# ─────────────────────────────────────────────
print("\n🔧 Encoding categorical variables...")

le = LabelEncoder()
cat_cols = ["job","marital","education","housing","loan","contact","poutcome"]
df_enc = df.copy()
for col in cat_cols:
    df_enc[col] = le.fit_transform(df_enc[col])

# Features & Target
X = df_enc.drop("purchased", axis=1)
y = df_enc["purchased"]

# Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   Train size : {X_train.shape[0]} samples")
print(f"   Test size  : {X_test.shape[0]} samples")


# ─────────────────────────────────────────────
# 3. TRAIN DECISION TREE
# ─────────────────────────────────────────────
print("\n🌳 Training Decision Tree Classifier...")

dt = DecisionTreeClassifier(
    max_depth=5,          # limit depth to avoid overfitting
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)
dt.fit(X_train, y_train)

# Predictions
y_pred = dt.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print(f"\n✅ Model Accuracy : {acc*100:.2f}%")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["No Purchase", "Purchase"]))


# ─────────────────────────────────────────────
# 4. FEATURE IMPORTANCE
# ─────────────────────────────────────────────
feat_imp = pd.Series(dt.feature_importances_,
                     index=X.columns).sort_values(ascending=False)
print("🔑 Top Features:")
print(feat_imp.head(5).to_string())


# ─────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#3a3a5c",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#2a2a4a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
})

fig = plt.figure(figsize=(20, 16))
fig.patch.set_facecolor("#0f0f1a")
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])   # Purchase distribution
ax2 = fig.add_subplot(gs[0, 1])   # Feature importance
ax3 = fig.add_subplot(gs[1, 0])   # Confusion matrix
ax4 = fig.add_subplot(gs[1, 1])   # Decision tree (shallow)


# ── Plot 1: Purchase Distribution ────────────
counts = df["purchased"].value_counts()
bars = ax1.bar(["No Purchase", "Purchase"],
               counts.values,
               color=["#ef5350", "#66bb6a"],
               edgecolor="#0f0f1a", linewidth=1.5, width=0.5)
for bar, val in zip(bars, counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 5,
             f"{val}\n({val/n*100:.1f}%)",
             ha="center", fontsize=11, fontweight="bold", color="white")
ax1.set_title("Purchase Distribution", fontsize=13, fontweight="bold", pad=10)
ax1.set_ylabel("Count")
ax1.set_ylim(0, counts.max() * 1.2)
ax1.grid(axis="y")


# ── Plot 2: Feature Importance ────────────────
colors = plt.cm.plasma(np.linspace(0.2, 0.9, len(feat_imp)))
bars = ax2.barh(feat_imp.index[::-1], feat_imp.values[::-1],
                color=colors[::-1], edgecolor="#0f0f1a", linewidth=0.8)
for bar, val in zip(bars, feat_imp.values[::-1]):
    ax2.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=9, color="white")
ax2.set_title("Feature Importance", fontsize=13, fontweight="bold", pad=10)
ax2.set_xlabel("Importance Score")
ax2.grid(axis="x")


# ── Plot 3: Confusion Matrix ──────────────────
cm = confusion_matrix(y_test, y_pred)
im = ax3.imshow(cm, cmap="YlOrRd", aspect="auto")
ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
ax3.set_xticklabels(["No Purchase", "Purchase"])
ax3.set_yticklabels(["No Purchase", "Purchase"])
ax3.set_xlabel("Predicted Label", fontsize=11)
ax3.set_ylabel("True Label", fontsize=11)
for i in range(2):
    for j in range(2):
        ax3.text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 fontsize=20, fontweight="bold",
                 color="white" if cm[i,j] > cm.max()/2 else "black")
ax3.set_title(f"Confusion Matrix  |  Accuracy: {acc*100:.1f}%",
              fontsize=13, fontweight="bold", pad=10)
plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)


# ── Plot 4: Decision Tree (depth=3 for clarity) ──
dt_vis = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_vis.fit(X_train, y_train)
plot_tree(dt_vis,
          feature_names=X.columns.tolist(),
          class_names=["No Buy", "Buy"],
          filled=True, rounded=True,
          fontsize=7, ax=ax4,
          impurity=False,
          proportion=True)
ax4.set_title("Decision Tree Structure (depth=3)",
              fontsize=13, fontweight="bold", pad=10)
ax4.set_facecolor("#1a1a2e")


# ── Super title ───────────────────────────────
fig.suptitle(
    "PRODIGY INFOTECH  |  Task-03  —  Decision Tree Classifier: Bank Marketing",
    fontsize=15, fontweight="bold", color="white", y=0.98
)

plt.savefig("prodigy_task03_output.png", dpi=150,
            bbox_inches="tight", facecolor=fig.get_facecolor())
print("\n✅ Plot saved as prodigy_task03_output.png")
plt.show()
