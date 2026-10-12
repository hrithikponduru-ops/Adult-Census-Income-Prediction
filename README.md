# Adult Census Income: Exploratory Data Analysis & Machine Learning

A Python project that explores the **Adult Census Income** dataset and compares machine-learning classifiers for predicting whether an individual's annual income is **`>50K`** or **`<=50K`**. It produces colorful exploratory visualizations, model evaluation charts, CSV summaries, and a serialized best-performing model.

## Features

- Cleans the dataset by trimming whitespace, treating `?` as missing, normalizing income labels, and removing duplicate rows.
- Generates **12 exploratory data analysis (EDA) figures** covering class balance, numeric and categorical distributions, income patterns, correlations, and demographic comparisons.
- Trains and evaluates **10 classification models** with consistent preprocessing and a stratified train/test split.
- Compares accuracy, precision, recall, F1 score, and ROC-AUC; also searches for an F1-maximizing classification threshold.
- Exports **5 model-related figures**, a model comparison CSV, summary statistics, and the top model (selected by ROC-AUC).

## Requirements

- Python 3.9+ recommended
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn` (version supporting `errorbar=`)
- `scikit-learn` (version supporting `OneHotEncoder(sparse_output=...)`)
- `xgboost`
- `lightgbm`
- `joblib`

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm joblib
```

## Dataset

The script expects an **Adult Census Income CSV** with 15 columns in this order:

```text
age, workclass, fnlwgt, education, education_num,
marital_status, occupation, relationship, race, sex,
capital_gain, capital_loss, hours_per_week, native_country, income
```

The usual headerless Adult dataset is supported; a file with a header row is also handled. Income labels should be `<=50K` and `>50K` (trailing periods are stripped automatically). Missing categorical values may be represented by `?`.

**Important:** The input CSV is not included with this script. Obtain the Adult dataset separately and place it at `input_data/adult.csv`, or pass its location as a command-line argument.

## Quick start

Suggested project layout:

```text
project/
├── adult_analysis.py
├── input_data/
│   └── adult.csv
└── output_data/          # created automatically when the script runs
```

Run with the default paths:

```bash
python adult_analysis.py
```

Or specify your input file and output directory:

```bash
python adult_analysis.py path/to/adult.csv path/to/results
```

**Arguments**

| Argument | Default | Description |
| --- | --- | --- |
| `Input_file` | `input_data/adult.csv` | Path to the Adult CSV dataset |
| `Output_dir` | `output_data/adult_results` | Directory for generated outputs |

The script creates the output directory if necessary, prints data-quality information and model scores, and writes all generated files there. Plots are saved directly to disk using Matplotlib's noninteractive `Agg` backend.

## Workflow

### 1. Data loading and cleaning

1. Read the dataset and assign the 15 expected column names.
2. Replace `?` with missing values and trim whitespace from text fields.
3. Remove trailing periods from income labels and drop duplicate rows.
4. Encode the target as `1` for `>50K` and `0` otherwise.
5. Export descriptive statistics to `summary_statistics.csv`.

### 2. Exploratory data analysis

The EDA examines income class balance, distributions of numerical features, category frequencies, income rates by group, demographic patterns, missingness, capital gains/losses, and correlations. It also creates a pairplot using a reproducible random sample of **3,000 rows**.

### 3. Machine-learning preprocessing

- **Numerical features:** `age`, `fnlwgt`, `education_num`, `capital_gain`, `capital_loss`, `hours_per_week`.
- **Categorical features:** `workclass`, `education`, `marital_status`, `occupation`, `relationship`, `race`, `sex`, `native_country`.
- `native_country` is reduced to `United-States` versus `Other`.
- Missing categorical values are filled with `Unknown`.
- `capital_gain` and `capital_loss` receive a `log1p` transformation.
- Numerical variables are standardized; categorical variables are one-hot encoded with unknown categories ignored.
- Data is split into **80% training / 20% testing**, stratified by the target, with random seed `42`.

### 4. Model training and evaluation

The script fits the following classifiers:

1. Logistic Regression
2. Gaussian Naive Bayes
3. K-Nearest Neighbors (KNN)
4. Decision Tree
5. Random Forest
6. AdaBoost
7. Gradient Boosting
8. XGBoost
9. LightGBM
10. Multi-Layer Perceptron (MLP)

For each model, the script reports **accuracy, precision, recall, F1, and ROC-AUC** using a default probability threshold of `0.5`. It also tests thresholds from `0.20` through `0.70` to find the highest test-set F1 score. The model with the highest **ROC-AUC** is saved as `best_model.joblib`.

## Generated outputs

All files are written to the configured output directory.

| File | Description |
| --- | --- |
| `summary_statistics.csv` | Descriptive statistics for dataset columns |
| `01_target_distribution.png` | Income class counts and proportions |
| `02_numeric_distributions.png` | Histograms of numerical variables |
| `03_numeric_by_income_kde.png` | Numerical density plots by income |
| `04_boxplots.png` | Numerical features compared across income classes |
| `05_categorical_counts.png` | Categorical feature counts by income |
| `06_high_income_rate_by_category.png` | Percentage earning >50K by category |
| `07_age_hours_education_insights.png` | Income rates by age, hours worked, and education |
| `08_sex_race_analysis.png` | Income patterns by sex, race, and age |
| `09_occupation_education_heatmap.png` | Income rates by occupation and education level |
| `10_correlation_heatmap.png` | Correlations between numerical features and target |
| `11_pairplot.png` | Pairwise plots of selected numerical features |
| `12_missing_and_capital.png` | Missing-value counts and capital gain/loss rates |
| `model_comparison.csv` | Metrics and F1 threshold results for all models |
| `13_model_comparison.png` | ROC-AUC and other model metrics |
| `14_roc_curves.png` | ROC curves for all 10 models |
| `15_confusion_matrices.png` | Confusion matrices at threshold 0.5 |
| `16_feature_importance.png` | Top 20 feature importances from the highest-ranked eligible tree model |
| `17_logreg_coefficients.png` | Strongest positive and negative logistic regression coefficients |
| `best_model.joblib` | Serialized pipeline for the model with the highest ROC-AUC |

## Loading the saved model

The saved artifact contains the fitted preprocessing pipeline and classifier. For inference, provide a DataFrame with the same **14 predictor columns**, after applying the script's pre-pipeline transformations (country grouping, missing categorical value handling, and `log1p` for capital variables):

```python
import joblib

model = joblib.load("output_data/adult_results/best_model.joblib")

# X_new must contain the same 14 predictor columns and the
# pre-pipeline transformations described above.
probabilities = model.predict_proba(X_new)[:, 1]
predictions = (probabilities >= 0.5).astype(int)
```

**Note:** The saved pipeline does **not** include the manual transformations applied before training. Applying the pipeline directly to raw CSV rows without those transformations may produce incorrect predictions.

## Limitations and considerations

- **Threshold selection uses the test set.** The script chooses each model's best-F1 threshold based on test labels. Those optimized F1 scores should not be treated as unbiased final performance estimates; use a separate validation set or cross-validation for threshold tuning.
- **Class imbalance:** The code calculates a positive-class weight but does not use it in model fitting. ROC-AUC and F1 help complement accuracy when classes are imbalanced.
- **Dataset size:** The pairplot requires at least 3,000 rows because it samples without replacement.
- **Sensitive attributes:** `race` and `sex` are used as predictors and in exploratory comparisons. Predictions may reflect historical biases in the underlying data; evaluate fairness before any consequential use.
- **Model persistence:** Only load `.joblib` files from trusted sources.
- **No model results are claimed here:** Actual performance depends on the supplied dataset and execution environment.

## Reproducibility

The script sets `RANDOM_STATE = 42` for its train/test split, pairplot sampling, and supported estimators. Library versions and execution environments can still affect results.

## License

No license is specified in the provided source code. Add a license file if you plan to distribute the project.
