# Pan-genome-based Feature Selection Approach to Explore Multi-Drug-Resistance Genes and Potential Biomarkers in *Klebsiella Pneumonia*

## Table of Contents
- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Data Format](#data-format)
- [Methods](#methods)
  - [Baseline: All Genes, All Models](#baseline-all-genes-all-models)
  - [Feature Selection: All Selectors, All Models](#feature-selection-all-selectors-all-models)
  - [Incremental AdaBoost Feature Selection (Main Pipeline)](#incremental-adaboost-feature-selection-main-pipeline)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Evaluation Metrics](#evaluation-metrics)


## Overview

This repository contains the machine learning pipeline accompanying our project on predicting antimicrobial resistance (AMR) phenotypes from bacterial genomic data. The pipeline takes gene presence/absence binary matrices as input and trains classifiers to distinguish resistant (R) from susceptible (S) isolates.

Three progressive experimental stages are implemented:

1. **Baseline evaluation** — training on all genes without feature selection across four classifier architectures.
2. **Feature selector comparison** — evaluating AdaBoost- and XGBoost-based feature selectors paired with multiple classifiers.
3. **Incremental AdaBoost selection** *(primary method)* — an incremental feature addition strategy that identifies the minimal optimal gene subset ranked by AdaBoost feature importances, evaluated with HistGradientBoosting classifier.

The incremental approach was selected as the final method because it systematically identifies the smallest feature set that maximises predictive performance, making the selected genes more interpretable and biologically meaningful. 

## Repository Structure

```
.
├── K.pneumonia.data.tar.gz/                 # Compressed data directory (TSV files)
│   └── <drug_name>
│
├── train_adaboost_incremental.py            # PRIMARY: Incremental AdaBoost feature selection
├── train_all_genes_all_models.py            # Baseline: all genes, four classifiers
├── train_all_selectors_all_models.py        # Selector comparison: AdaBoost vs XGBoost
│
└── README.md
```

## Data Format

Each input file is a **tab-separated (TSV) file** representing one antibiotic drug. Files are compressed and placed in the `K.pneumonia.data.tar.gz/` directory. Each row is a bacterial isolate; each column after the metadata columns is a gene.

| Column | Type | Description |
|---|---|---|
| `genome_id` | string | Unique identifier for the bacterial isolate |
| `resistant_phenotype` | string | Phenotypic label: `R` (resistant) or `S` (susceptible) |
| `gene_*` | integer (0/1) | Binary gene presence (1) / absence (0) |

**Example:**

```
genome_id       resistant_phenotype     gene_1  gene_10  gene_1000  gene_100013
573.129         S                       0       0        1          0
573.12961       R                       0       0        1          0
573.1298        S                       0       0        1          0
573.13002       S                       0       0        1          0
573.1289        S                       0       0        1          0
```

> **Note:** The pipeline is designed to process multiple drugs simultaneously. Extract data and place all drug TSV files to a local `data/` directory and each will be processed independently.

## Methods

### Baseline: All Genes, All Models

**Script:** `train_all_genes_all_models.py`

Trains four classifiers on the complete gene feature space without any feature selection, establishing a performance ceiling and baseline reference. Uses 10-fold stratified cross-validation. **It is noted that this process consume a lot of time to finish due to high dimentional data.** 

**Classifiers:**

| Key | Classifier | Key Hyperparameters |
|---|---|---|
| `SVC` | Support Vector Classifier | `kernel='linear'`, `C=0.1`, `class_weight='balanced'` |
| `LogReg` | Logistic Regression | `max_iter=10000`, `class_weight='balanced'` |
| `RF` | Random Forest | `class_weight='balanced'` |
| `HistGrad` | HistGradientBoosting Classifier | `class_weight='balanced'` |

---

### Feature Selection: All Selectors, All Models

**Script:** `train_all_selectors_all_models.py`

Evaluates two tree-based feature selectors paired with all four classifiers above. Each selector is fitted on training fold data and transforms both train and test splits before model training. Uses 10-fold stratified cross-validation.

**Feature Selectors:**

| Key | Selector |
|---|---|
| `AdaBoost` | `SelectFromModel(AdaBoostClassifier(algorithm='SAMME'))` |
| `XGBoost` | `SelectFromModel(XGBClassifier())` |

---

### Incremental AdaBoost Feature Selection (Main Pipeline)

**Script:** `train_adaboost_incremental.py`

This is the **primary method** reported in our publication. It searches for the *optimal number* of features rather than using a fixed threshold.

**Cross-validation:** 10-fold Stratified K-Fold (`random_state=42`)

**Schematic:**

```
For each CV fold:
  ┌─────────────────────────────────────────────┐
  │ 1. Fit AdaBoostClassifier on X_train        │
  │ 2. Rank genes by feature_importances_       │
  │ 3. For k = 1 to N:                          │
  │      Select top-k genes                     │
  │      Fit HistGradientBoosting               │
  │      Evaluate AUC on X_test                 │
  │ 4. Record best k and its metrics            │
  └─────────────────────────────────────────────┘
Aggregate fold-level best metrics → mean ± std
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Kai-PN/K.pneumonia-AMR.git
cd <your_repo>

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows

# Install dependencies (required Python 3.9)
pip install scikit-learn==1.5.1 pandas==2.2.3 numpy==1.22.4 joblib==1.4.2 xgboost==2.1.3 
```

## Usage

Extract data in the `K.pneumonia.data.tar.gz` and place all input TSV files (one per antibiotic) in a local `data/` directory, then run the desired script from the repository root.

### Example: Primary pipeline (`adaboost_incremental/`)
```bash
python train_adaboost_incremental.py
```

> **Parallelism:** All scripts use `joblib.Parallel`. Adjust `n_jobs` in each script to match the number of CPU cores available on your system. The default values are `n_jobs=2` (incremental), `n_jobs=20` (baseline), and `n_jobs=10` (selector comparison).

## Output Files

### Example: Primary pipeline (`adaboost_incremental/`)

| File | Description |
|---|---|
| `adaboost.incremental.detailed.txt` | All metrics for every (fold × feature count) combination |
| `adaboost.incremental.results.txt` | Best metrics per fold (the fold-level optimal feature count) |
| `adaboost.incremental.mean.txt` | Mean of per-fold best metrics, grouped by Drug / Model / Selector |
| `adaboost.incremental.std.txt` | Standard deviation of per-fold best metrics |
| `adaboost.incremental.best.folds.list.txt` | The single best fold per group with its selected gene list |

## Evaluation Metrics

All models are evaluated with the following metrics computed on the held-out test fold:

| Metric | Description |
|---|---|
| **AUC** | Area Under the ROC Curve — primary selection criterion |
| **Accuracy** | Overall classification accuracy |
| **Precision** | Positive predictive value for the resistant class |
| **Recall** | Sensitivity / true positive rate for the resistant class |
| **F1** | Harmonic mean of Precision and Recall |
| **MCC** | Matthews Correlation Coefficient — robust to class imbalance |

## License

This project is licensed under the MIT License. See `LICENSE` for details.s