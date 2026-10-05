# CreditGuard — Credit Risk & Default Prediction

CreditGuard is an end-to-end machine learning project for predicting whether a borrower will experience **serious delinquency within two years**. The project uses the **Give Me Some Credit** dataset and focuses on building an interpretable, leakage-safe credit-risk pipeline using **Logistic Regression** and **Linear SVM**.

The goal is not only to classify borrowers, but also to convert model outputs into useful **default probabilities and risk segments**.

---

## Project Overview

Credit-risk datasets are often highly imbalanced, which makes raw accuracy misleading. In this dataset, only about **6.68%** of borrowers belong to the positive/default class.

This project therefore focuses on:

- Data validation and exploratory analysis
- Missing-value and outlier handling
- Interpretable feature engineering
- Leakage-safe train/validation/test splitting
- Dummy baseline, Logistic Regression and Linear SVM
- ROC-AUC, PR-AUC, Precision, Recall and F1 evaluation
- Validation-based classification-threshold tuning
- Default-probability estimation
- Borrower risk segmentation
- Business interpretation of model outputs

---

## Project Architecture

```text
                    RAW DATA
                       │
                       ▼
              Data Validation
                       │
                       ▼
                     EDA
          ┌────────────┴───────────┐
          │                        │
    Missing Values             Outliers
          │                        │
          └────────────┬───────────┘
                       ▼
              Feature Engineering
                       │
                       ▼
      Stratified Train / Validation / Test
                       │
                       ▼
                ML Pipeline
                       │
          ┌────────────┼─────────────┐
          ▼            ▼             ▼
       Dummy       Logistic       Linear
     Classifier    Regression       SVM
                       │
                       ▼
              Model Comparison
                       │
                       ▼
      ROC-AUC / PR-AUC / Recall / F1
                       │
                       ▼
              Threshold Tuning
                       │
                       ▼
              Default Probability
                       │
                       ▼
               Risk Segmentation
                       │
                       ▼
           Business Interpretation
```

---

## Dataset

**Dataset:** Give Me Some Credit  
**Training observations:** 150,000 borrowers  
**Target:** `SeriousDlqin2yrs`

- `0` — borrower did not experience serious delinquency within two years
- `1` — borrower experienced serious delinquency within two years

### Target distribution

| Class | Borrowers | Share |
|---|---:|---:|
| No serious delinquency | 139,974 | 93.32% |
| Serious delinquency | 10,026 | 6.68% |

Because the positive class is rare, **accuracy is not used as the primary model-selection metric**.

### Missing values

The two main features containing missing values are:

| Feature | Missing values |
|---|---:|
| `MonthlyIncome` | 29,731 |
| `NumberOfDependents` | 3,924 |

Median imputation is applied inside the modelling pipeline so that imputation statistics are learned only from training data.

---

## Data Cleaning

The workflow performs several transparent data-quality checks before modelling:

- Removes the CSV row identifier from the feature set
- Treats `age <= 0` as invalid and converts it to missing
- Treats repeated delinquency-count values `96` and `98` as anomalous coded values
- Preserves duplicate feature profiles because different borrowers can legitimately share identical observed characteristics
- Caps extreme tails of selected continuous variables using limits learned from the training set only

This keeps preprocessing leakage-safe and avoids using information from validation or test data when defining transformations.

---

## Feature Engineering

Only a few interpretable features are added:

### Total Delinquencies

```text
TotalDelinquencies =
30–59 day delinquencies
+ 60–89 day delinquencies
+ 90+ day delinquencies
```

### Income per Dependent

```text
IncomePerDependent = MonthlyIncome / (NumberOfDependents + 1)
```

### Real-Estate Loan Indicator

```text
HasRealEstateLoan = 1 if borrower has at least one real-estate loan/line
                    0 otherwise
```

The objective is to improve interpretability without creating an unnecessarily complex feature set.

---

## Train / Validation / Test Strategy

The labelled dataset is split using **stratified sampling**:

- **70% Training**
- **15% Validation**
- **15% Test**

Stratification preserves the rare-event class ratio across all three sets.

The validation set is used for **classification-threshold selection**. The test set is kept untouched until final evaluation, preventing threshold-selection leakage.

---

## Models

### 1. Dummy Classifier

A majority-class baseline is included to show why accuracy can be misleading on imbalanced data.

A classifier predicting every borrower as non-default can achieve high accuracy while detecting **zero actual defaulters**.

### 2. Logistic Regression

Logistic Regression is the main risk-scoring model because it is:

- Fast
- Interpretable
- Suitable for large tabular datasets
- Able to output estimated default probabilities
- Easy to convert into borrower risk bands

The pipeline is:

```text
Median Imputation → StandardScaler → Logistic Regression
```

### 3. Linear SVM

Linear SVM is used as a maximum-margin comparison model.

```text
Median Imputation → StandardScaler → Linear SVM
```

The linear variant is used because the dataset is large and relatively low-dimensional. A kernel SVM would add significant computational cost without being necessary for the purpose of this project.

Unlike Logistic Regression, standard Linear SVM produces a **decision score rather than a calibrated default probability**, so Logistic Regression remains the final risk-scoring model.

---

## Model Evaluation

Models are evaluated using metrics appropriate for an imbalanced classification problem:

- **ROC-AUC** — how well the model ranks risky borrowers above safer borrowers across thresholds
- **PR-AUC / Average Precision** — particularly useful when the positive class is rare
- **Precision** — among borrowers flagged as risky, how many actually defaulted
- **Recall** — among actual defaulters, how many were detected
- **F1 Score** — harmonic balance between precision and recall
- **Balanced Accuracy** — gives equal importance to both classes

### ROC-AUC Results

| Model | ROC-AUC |
|---|---:|
| Dummy Classifier | **0.500** |
| Logistic Regression | **0.850** |
| Linear SVM | **0.856** |

The two learned models provide a substantial improvement over random/majority-class ranking. Linear SVM achieves the highest ROC-AUC, while Logistic Regression is retained as the final risk-scoring model because it naturally produces interpretable probabilities.

![ROC Curves](assets/roc_curve.png)

---

## Threshold Tuning

A default classification threshold of `0.50` is not necessarily appropriate for credit risk.

For example:

- **False negative:** an actual defaulter is classified as safe
- **False positive:** a reliable borrower is classified as risky

The Logistic Regression threshold is therefore tuned using the **validation set only**, maximising validation F1.

### Selected validation threshold

| Metric | Value |
|---|---:|
| Threshold | **0.1693** |
| Precision | **0.4045** |
| Recall | **0.4734** |
| F1 Score | **0.4363** |

Lowering the threshold increases the model's willingness to flag borrowers as risky, which generally increases recall at the cost of precision.

---

## Credit Risk Segmentation

Logistic Regression produces an estimated probability of default for each borrower. These probabilities are converted into illustrative risk bands:

| Predicted probability | Risk band |
|---|---|
| `< 5%` | Low |
| `5–10%` | Moderate |
| `10–20%` | High |
| `>= 20%` | Very High |

These cut-offs are used for presentation and analysis. In a real lending environment, thresholds would be determined using expected financial loss, approval policies, regulation and probability-calibration analysis.

---

## Model Interpretability

Because Logistic Regression is trained on standardised features, its coefficients can be compared directionally:

- **Positive coefficient** → associated with higher predicted default risk
- **Negative coefficient** → associated with lower predicted default risk
- `exp(coefficient)` → odds multiplier for approximately a one-standard-deviation increase in that feature, holding other variables constant

These coefficients describe relationships within the fitted model and should **not** be interpreted as causal effects.

---

## Why Not Accuracy?

The dataset contains only **6.68% positive cases**.

A model that predicts every borrower as non-default would achieve roughly:

```text
93.32% accuracy
```

while having:

```text
Recall for defaulters = 0
```

For this reason, ROC-AUC, PR-AUC, Precision, Recall and F1 are more informative than raw accuracy.

---

## Business Interpretation

The final workflow addresses four practical questions:

1. **Can the model rank risky borrowers effectively?**  
   Evaluated using ROC-AUC and PR-AUC.

2. **How many actual defaulters are missed?**  
   Evaluated using Recall and the confusion matrix.

3. **What happens when the classification threshold changes?**  
   Threshold tuning exposes the trade-off between Precision and Recall.

4. **Can model probabilities be converted into actionable groups?**  
   Borrowers are segmented into Low, Moderate, High and Very High predicted-risk bands.

The result is not just a binary classifier but an interpretable **credit-risk scoring workflow**.

---

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Google Colab / Jupyter Notebook

---

## Repository Structure

```text
CreditGuard/
│
├── README.md
├── CreditGuard_Credit_Risk_Project_Colab.ipynb
│
├── data/
│   └── README.md
│
├── assets/
│   └── roc_curve.png
│
└── outputs/
    ├── model_metrics.csv
    ├── threshold_metrics.csv
    ├── threshold_comparison.csv
    ├── risk_scores.csv
    ├── risk_band_summary.csv
    └── logistic_feature_effects.csv
```

The original Kaggle data should generally not be committed directly if dataset-distribution terms do not permit redistribution. The notebook can instead prompt the user to provide `cs-training.csv`.

---

## How to Run

1. Open `CreditGuard_Credit_Risk_Project_Colab.ipynb` in Google Colab.
2. Run all cells.
3. Upload `cs-training.csv` when prompted.
4. The notebook will perform preprocessing, model training, evaluation, threshold tuning and risk segmentation.
5. Generated outputs are saved under `creditguard_outputs/` and packaged as `creditguard_outputs.zip`.

---

## Key Takeaways

- Credit default is a **highly imbalanced classification problem**, so accuracy alone is misleading.
- Logistic Regression achieved **0.850 ROC-AUC** while remaining interpretable and probability-based.
- Linear SVM achieved a slightly higher **0.856 ROC-AUC**, demonstrating strong ranking performance.
- Threshold tuning produced a validation threshold of **0.1693**, explicitly balancing Precision and Recall.
- The final Logistic Regression model converts borrower characteristics into **default probabilities and interpretable risk bands**.
- All preprocessing and threshold selection are performed using training/validation data to keep the final test evaluation leakage-safe.

---

## Interview Summary

> Built a leakage-safe credit-risk prediction pipeline on 150,000 borrowers from the Give Me Some Credit dataset. Performed data validation, missing-value and outlier handling, interpretable feature engineering and stratified train/validation/test splitting. Compared a Dummy baseline, Logistic Regression and Linear SVM using ROC-AUC, PR-AUC, Precision, Recall and F1, achieving ROC-AUC scores of 0.850 and 0.856 for Logistic Regression and Linear SVM respectively. Tuned the Logistic Regression classification threshold on a separate validation set and converted predicted default probabilities into business-friendly borrower risk bands.

---

## Disclaimer

This project is for educational and portfolio purposes only. It is not intended to be used for real-world lending, credit approval or financial decision-making.
