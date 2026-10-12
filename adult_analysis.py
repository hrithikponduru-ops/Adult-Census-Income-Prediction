"""
Adult Census Income  -  EDA + Machine Learning
------------------------------------------------
Usage:  python adult_analysis.py Input_file Output_dir
Part 1: Exploratory Data Analysis (bright colour palette)
Part 2: Model training, comparison and evaluation
"""
import os
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

CSV = sys.argv[1] if len(sys.argv) > 1 else "input_data/adult.csv"
OUT = sys.argv[2] if len(sys.argv) > 2 else "output_data/adult_results"
os.makedirs(OUT, exist_ok=True)
RANDOM_STATE = 42

# --------------------------------------------------------------------------
# Bright theme
# --------------------------------------------------------------------------
BRIGHT = ["#FF1744", "#00E5FF", "#FFEA00", "#76FF03", "#D500F9", "#FF9100", "#2979FF", "#1DE9B6", "#F50057", "#C6FF00"]
C_LOW, C_HIGH = "#00B0FF", "#FF1744"        # <=50K (electric blue), >50K (hot red)
INCOME_PAL = {"<=50K": C_LOW, ">50K": C_HIGH}
sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"axes.titleweight": "bold", "axes.titlesize": 14, "figure.dpi": 110, "savefig.bbox": "tight"})
bright_cmap = sns.blend_palette(["#00E5FF", "#FFFFFF", "#FF1744"], as_cmap=True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=130)
    plt.close(fig)
    print("saved", name)


# --------------------------------------------------------------------------
# Load & clean
# --------------------------------------------------------------------------
cols = ["age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country",
        "income"]
# The file has no header row, so supply names (and keep the first data row)
df = pd.read_csv(CSV, header=None, names=cols, skipinitialspace=True)
# If a header *was* present in a different file, drop it
if df["age"].dtype == object:
    df = pd.read_csv(CSV, skipinitialspace=True)
    df.columns = cols

df = df.replace("?", np.nan)
for c in df.select_dtypes(include=["object", "string"]).columns:
    df[c] = df[c].str.strip()
df["income"] = df["income"].str.replace(".", "", regex=False)

print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
n_dups = df.duplicated().sum()
print("Duplicate rows:", n_dups)
df = df.drop_duplicates().reset_index(drop=True)

df["target"] = (df["income"] == ">50K").astype(int)
num_cols = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
cat_cols = ["workclass", "education", "marital_status", "occupation", "relationship", "race", "sex", "native_country"]

df.describe(include="all").T.to_csv(os.path.join(OUT, "summary_statistics.csv"))

# ==========================================================================
# PART 1 -  EDA
# ==========================================================================
# 1. Target distribution --------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
counts = df["income"].value_counts()
sns.countplot(data=df, x="income", palette=INCOME_PAL, order=["<=50K", ">50K"], ax=ax[0])
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()),
                   ha="center", va="bottom", fontweight="bold")
ax[0].set_title("Income Class Counts")
ax[1].pie(counts[["<=50K", ">50K"]], labels=["<=50K", ">50K"], autopct="%1.1f%%",
          colors=[C_LOW, C_HIGH], startangle=90, explode=(0, 0.08),
          wedgeprops=dict(edgecolor="white", linewidth=3),
          textprops=dict(fontweight="bold", fontsize=13))
ax[1].set_title("Class Balance (imbalanced ~3:1)")
save(fig, "01_target_distribution.png")

# 2. Numeric distributions ------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, c, col in zip(axes.ravel(), num_cols, BRIGHT):
    sns.histplot(df[c], bins=40, kde=True, color=col, edgecolor="white", ax=ax)
    ax.set_title(f"Distribution of {c}")
fig.suptitle("Numeric Feature Distributions", fontsize=18, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "02_numeric_distributions.png")

# 3. Numeric by income (KDE) ---------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(17, 9))
for ax, c in zip(axes.ravel(), num_cols):
    data = df if c not in ("capital_gain", "capital_loss") else df[df[c] > 0]
    sns.kdeplot(data=data, x=c, hue="income", palette=INCOME_PAL, fill=True,
                alpha=0.5, common_norm=False, linewidth=2.5, ax=ax)
    ax.set_title(f"{c} by income" + (" (non-zero only)" if c in ("capital_gain", "capital_loss") else ""))
fig.suptitle("Numeric Features Split by Income", fontsize=18, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "03_numeric_by_income_kde.png")

# 4. Box plots -------------------------------------------------------------
fig, axes = plt.subplots(1, 4, figsize=(18, 5.5))
for ax, c in zip(axes, ["age", "education_num", "hours_per_week", "fnlwgt"]):
    sns.boxplot(data=df, x="income", y=c, palette=INCOME_PAL, order=["<=50K", ">50K"],
                linewidth=2, fliersize=2.5, ax=ax)
    ax.set_title(f"{c} vs income")
fig.tight_layout()
save(fig, "04_boxplots.png")

# 5. Categorical counts ------------------------------------------------------
fig, axes = plt.subplots(4, 2, figsize=(17, 24))
for ax, c in zip(axes.ravel(), cat_cols):
    order = df[c].value_counts().index[:12]
    sns.countplot(data=df[df[c].isin(order)], y=c, order=order, hue="income",
                  palette=INCOME_PAL, hue_order=["<=50K", ">50K"], ax=ax)
    ax.set_title(f"{c} (top {len(order)})")
    ax.set_xscale("log") if c in ("native_country", "race") else None
fig.suptitle("Categorical Features by Income", fontsize=20, fontweight="bold", y=1.0)
fig.tight_layout()
save(fig, "05_categorical_counts.png")

# 6. % earning >50K by category ---------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(19, 11))
for ax, c in zip(axes.ravel(), ["education", "occupation", "marital_status", "workclass", "relationship", "race"]):
    rate = (df.groupby(c)["target"].mean() * 100).sort_values()
    colors = sns.color_palette("turbo", len(rate))
    ax.barh(rate.index, rate.values, color=colors, edgecolor="white")
    ax.axvline(df["target"].mean() * 100, color="#111", ls="--", lw=2, label="overall")
    ax.set_title(f"% earning >50K by {c}")
    ax.set_xlabel("% >50K")
    ax.legend()
fig.tight_layout()
save(fig, "06_high_income_rate_by_category.png")

# 7. Age / hours / education insights ---------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
df["age_group"] = pd.cut(df["age"], [16, 25, 35, 45, 55, 65, 100],
                         labels=["17-25", "26-35", "36-45", "46-55", "56-65", "65+"])
r = df.groupby("age_group", observed=True)["target"].mean() * 100
axes[0].bar(r.index.astype(str), r.values, color=BRIGHT[:6], edgecolor="white", linewidth=2)
axes[0].set_title("% >50K by Age Group"); axes[0].set_ylabel("% >50K")
for i, v in enumerate(r.values):
    axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

df["hours_group"] = pd.cut(df["hours_per_week"], [0, 20, 35, 40, 50, 60, 100],
                           labels=["<=20", "21-35", "36-40", "41-50", "51-60", "60+"])
r = df.groupby("hours_group", observed=True)["target"].mean() * 100
axes[1].bar(r.index.astype(str), r.values, color=BRIGHT[4:10], edgecolor="white", linewidth=2)
axes[1].set_title("% >50K by Weekly Hours"); axes[1].set_ylabel("% >50K")
for i, v in enumerate(r.values):
    axes[1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

r = df.groupby("education_num")["target"].mean() * 100
axes[2].plot(r.index, r.values, marker="o", markersize=10, lw=3.5, color="#FF1744",
             markerfacecolor="#FFEA00", markeredgecolor="#111", markeredgewidth=1.5)
axes[2].fill_between(r.index, r.values, alpha=0.2, color="#FF1744")
axes[2].set_title("% >50K by Years of Education"); axes[2].set_xlabel("education_num")
fig.tight_layout()
save(fig, "07_age_hours_education_insights.png")

# 8. Gender / race analysis -----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
sns.barplot(data=df, x="sex", y="target", palette=["#F50057", "#2979FF"], ax=axes[0], errorbar=None, edgecolor="white", linewidth=2)
axes[0].set_title("P(>50K) by Sex"); axes[0].set_ylabel("proportion >50K")
sns.barplot(data=df, x="race", y="target", palette=BRIGHT[:5], ax=axes[1], errorbar=None, edgecolor="white", linewidth=2)
axes[1].set_title("P(>50K) by Race"); axes[1].tick_params(axis="x", rotation=20)
sns.lineplot(data=df, x="age", y="target", hue="sex", palette=["#F50057", "#2979FF"], lw=3, ax=axes[2], errorbar=None)
axes[2].set_title("P(>50K) by Age and Sex")
fig.tight_layout()
save(fig, "08_sex_race_analysis.png")

# 9. Heatmap occupation x education level --------------------------------------
fig, ax = plt.subplots(figsize=(14, 8))
pivot = df.pivot_table(index="occupation", columns="education_num", values="target", aggfunc="mean") * 100
sns.heatmap(pivot, cmap="turbo", annot=True, fmt=".0f", linewidths=.5, ax=ax, cbar_kws={"label": "% earning >50K"}, annot_kws={"size": 8})
ax.set_title("% Earning >50K : Occupation x Years of Education")
save(fig, "09_occupation_education_heatmap.png")

# 10. Correlation heatmap --------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7.5))
corr = df[num_cols + ["target"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm_r", vmin=-1, vmax=1,
            linewidths=2, linecolor="white", square=True, ax=ax,
            annot_kws={"fontweight": "bold"})
ax.set_title("Correlation Matrix (numeric features + target)")
save(fig, "10_correlation_heatmap.png")

# 11. Pairplot ------------------------------------------------------------------------
sample = df.sample(3000, random_state=RANDOM_STATE)
g = sns.pairplot(sample, vars=["age", "education_num", "hours_per_week", "capital_gain"],
                 hue="income", palette=INCOME_PAL, plot_kws=dict(alpha=0.6, s=18),
                 diag_kind="kde", corner=True)
g.fig.suptitle("Pairplot (3,000-row sample)", y=1.02, fontsize=16, fontweight="bold")
save(g.fig, "11_pairplot.png")

# 12. Missing values + capital features --------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
miss = df[["workclass", "occupation", "native_country"]].isna().sum()
axes[0].bar(miss.index, miss.values, color=["#FF9100", "#D500F9", "#1DE9B6"],
            edgecolor="white", linewidth=2)
axes[0].set_title("Missing Values (originally '?')")
for i, v in enumerate(miss.values):
    axes[0].text(i, v + 20, f"{v:,}", ha="center", fontweight="bold")
cg = (df["capital_gain"] > 0).groupby(df["income"]).mean() * 100
axes[1].bar(cg.index, cg.values, color=[C_LOW, C_HIGH], edgecolor="white", linewidth=2)
axes[1].set_title("% with Capital Gain > 0")
cl = (df["capital_loss"] > 0).groupby(df["income"]).mean() * 100
axes[2].bar(cl.index, cl.values, color=[C_LOW, C_HIGH], edgecolor="white", linewidth=2)
axes[2].set_title("% with Capital Loss > 0")
fig.tight_layout()
save(fig, "12_missing_and_capital.png")

# ==========================================================================
# PART 2 -  MACHINE LEARNING
# ==========================================================================
print("\n" + "=" * 70 + "\nMACHINE LEARNING\n" + "=" * 70)
X = df[num_cols + cat_cols].copy()
# Simplify native_country to avoid 40+ sparse dummies
X["native_country"] = np.where(X["native_country"] == "United-States", "United-States", "Other")
X[cat_cols] = X[cat_cols].fillna("Unknown")
X["capital_gain"] = np.log1p(X["capital_gain"])
X["capital_loss"] = np.log1p(X["capital_loss"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
])
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
    "Decision Tree": DecisionTreeClassifier(max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1, random_state=RANDOM_STATE),
    "AdaBoost": AdaBoostClassifier(n_estimators=300, learning_rate=0.5, random_state=RANDOM_STATE), 
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=250, max_depth=4, learning_rate=0.1, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, verbose=-1),
    "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, early_stopping=True, max_iter=200, random_state=RANDOM_STATE),
}

results, probas, fitted = [], {}, {}
for name, clf in models.items():
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)
    p = pipe.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    # best-F1 threshold (useful because of 3:1 class imbalance)
    ths = np.linspace(0.2, 0.7, 51)
    best_th = ths[np.argmax([f1_score(y_test, (p >= t).astype(int)) for t in ths])]
    results.append(dict(Model=name,
                        Accuracy=accuracy_score(y_test, pred),
                        Precision=precision_score(y_test, pred),
                        Recall=recall_score(y_test, pred),
                        F1=f1_score(y_test, pred),
                        ROC_AUC=roc_auc_score(y_test, p),
                        Best_F1_Threshold=best_th,
                        F1_at_best_threshold=f1_score(y_test, (p >= best_th).astype(int))))
    probas[name], fitted[name] = p, pipe
    print(f"{name:22s} acc={results[-1]['Accuracy']:.4f}  auc={results[-1]['ROC_AUC']:.4f}  f1={results[-1]['F1']:.4f}")

res = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
res.round(4).to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
print("\n", res.round(4).to_string(index=False))
best_name = res.loc[0, "Model"]

# --- Model comparison bar chart --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
order = res["Model"]
sns.barplot(data=res, y="Model", x="ROC_AUC", order=order, palette="rainbow", ax=axes[0],
            edgecolor="white")
axes[0].set_xlim(0.7, 1.0); axes[0].set_title("ROC-AUC by Model")
for i, v in enumerate(res["ROC_AUC"]):
    axes[0].text(v + 0.002, i, f"{v:.3f}", va="center", fontweight="bold")
m = res.melt(id_vars="Model", value_vars=["Accuracy", "Precision", "Recall", "F1"])
sns.barplot(data=m, y="Model", x="value", hue="variable", order=order,
            palette=["#FF1744", "#00E5FF", "#FFEA00", "#76FF03"], ax=axes[1], edgecolor="white")
axes[1].set_title("Accuracy / Precision / Recall / F1 (threshold 0.5)")
axes[1].set_xlim(0.3, 1.0); axes[1].set_ylabel("")
fig.tight_layout()
save(fig, "13_model_comparison.png")

# --- ROC curves ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 7.5))
for (name, p), col in zip(probas.items(), BRIGHT):
    fpr, tpr, _ = roc_curve(y_test, p)
    ax.plot(fpr, tpr, lw=2.5, color=col, label=f"{name} ({roc_auc_score(y_test, p):.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1.5)
ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curves - All Models"); ax.legend(loc="lower right", fontsize=9)
save(fig, "14_roc_curves.png")

# --- Confusion matrices ------------------------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(24, 9))
for ax, (name, p) in zip(axes.ravel(), probas.items()):
    cm = confusion_matrix(y_test, (p >= 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="spring", cbar=False, linewidths=3,
                linecolor="white", annot_kws={"size": 15, "fontweight": "bold"},
                xticklabels=["<=50K", ">50K"], yticklabels=["<=50K", ">50K"], ax=ax)
    ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.tight_layout()
save(fig, "15_confusion_matrices.png")

# --- Feature importance (best tree model) -------------------------------------
tree_models = [n for n in res["Model"] if n in
               ("XGBoost", "LightGBM", "Random Forest", "Gradient Boosting")]
fi_name = tree_models[0]
pipe = fitted[fi_name]
names = pipe.named_steps["pre"].get_feature_names_out()
imp = pd.Series(pipe.named_steps["clf"].feature_importances_, index=names)
imp = (imp / imp.sum()).sort_values(ascending=False).head(20)
fig, ax = plt.subplots(figsize=(10, 8))
imp[::-1].plot.barh(ax=ax, color=sns.color_palette("turbo", 20), edgecolor="white")
ax.set_title(f"Top 20 Feature Importances ({fi_name})")
save(fig, "16_feature_importance.png")

# --- Logistic regression coefficients ------------------------------------------
lr = fitted["Logistic Regression"]
coef = pd.Series(lr.named_steps["clf"].coef_[0], index=lr.named_steps["pre"].get_feature_names_out())
top = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
fig, ax = plt.subplots(figsize=(10, 8))
ax.barh(top.index, top.values, color=np.where(top.values > 0, C_HIGH, C_LOW), edgecolor="white")
ax.set_title("Logistic Regression: Strongest Coefficients\n(red pushes toward >50K, blue toward <=50K)")
save(fig, "17_logreg_coefficients.png")

# --- Save best model --------------------------------------------------------------
import joblib
joblib.dump(fitted[best_name], os.path.join(OUT, "best_model.joblib"))
print(f"\nBest model by ROC-AUC: {best_name}  (saved to best_model.joblib)")
print("All outputs in:", OUT)
