
# Breast Cancer Classification & PCA

A machine learning project that compares classification algorithms and Principal Component Analysis (PCA) configurations on two Wisconsin Breast Cancer datasets. The project explores data preprocessing, dimensionality reduction, hyperparameter tuning, cross-validation, and model evaluation.

## Overview

The goal is to investigate how dimensionality reduction affects breast cancer classification performance and compare different machine learning models using multiple evaluation metrics.

The project uses two datasets:
- Wisconsin Diagnostic Breast Cancer (WDBC)
- Original Wisconsin Breast Cancer Dataset

## Objectives

- Perform exploratory data analysis on both datasets.
- Handle missing values and preprocess features.
- Standardize features before applying PCA.
- Compare models with and without dimensionality reduction.
- Tune selected model hyperparameters using GridSearchCV.
- Evaluate performance using multiple classification metrics.
- Compare the results obtained from both datasets.

## Technologies Used

- **Language:** Python
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Model Persistence:** Joblib
- **Environment:** Google Colab

## Datasets

### 1. Wisconsin Diagnostic Breast Cancer (WDBC)

- 569 samples
- 30 numerical features before removing the ID column
- Target variable: diagnosis (`M` = Malignant, `B` = Benign)

### 2. Original Wisconsin Breast Cancer Dataset

- 699 samples
- 9 predictive features
- Missing values represented by `?`
- Original class labels: `2` = Benign, `4` = Malignant

**Dataset source:** UCI Machine Learning Repository.

## Project Workflow

### 1. Exploratory Data Analysis

- Inspect dataset dimensions, data types, and descriptive statistics.
- Visualize missing values using heatmaps.
- Examine class distributions.
- Analyze feature correlations.
- Plot feature distributions and boxplots to inspect potential outliers.

### 2. Data Preprocessing

- Remove identifier columns.
- Convert diagnosis labels into binary classes.
- Handle missing values in the Original Wisconsin dataset.
- Split data into training and testing sets using stratification.
- Standardize numerical features using `StandardScaler`.

### 3. Principal Component Analysis

PCA is used to investigate the effect of dimensionality reduction on classification performance.

**WDBC configurations**
- No PCA: 30 features
- PCA: 15 components
- PCA: 10 components
- PCA: 5 components

**Original Wisconsin configurations**
- No PCA: 9 features
- PCA: 5 components
- PCA: 3 components
- PCA: 2 components

Cumulative explained variance plots are generated to examine how much variance is retained as the number of principal components changes.

### 4. Machine Learning Models

Four classification algorithms are compared:

| Model | Description |
|---|---|
| Logistic Regression | Linear classification baseline |
| Support Vector Machine (SVM) | Classification using an RBF kernel |
| Random Forest | Ensemble of decision trees |
| K-Nearest Neighbors (KNN) | Distance-based classification |

### 5. Hyperparameter Tuning

GridSearchCV with stratified 5-fold cross-validation is used for selected models.

| Model | Hyperparameters |
|---|---|
| SVM | `C`, `gamma`, `kernel` |
| Random Forest | `n_estimators`, `max_depth` |
| KNN | `n_neighbors` |

Logistic Regression uses its default configuration in the current experiment.

### 6. Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-validation score
- Train-test accuracy gap

The notebook also generates:
- Confusion matrices
- ROC curves
- Learning curves
- PCA explained variance plots
- Dataset comparison visualizations

## Results

### Original Wisconsin Breast Cancer Dataset

The best-ranked configuration in the supplied experiment table is Random Forest with 3 principal components.

| PCA Configuration | Model | Train Accuracy | Test Accuracy | CV Score | Overfitting Gap | Precision | Recall | F1-score | Final Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCA 3 | Random Forest | 1.0000 | 0.9714 | 0.9660 | 0.0286 | 0.9400 | 0.9792 | 0.9592 | 0.96778 |
| PCA 2 | Random Forest | 1.0000 | 0.9714 | 0.9643 | 0.0286 | 0.9400 | 0.9792 | 0.9592 | 0.96744 |
| PCA 2 | SVM | 0.9696 | 0.9643 | 0.9714 | 0.0053 | 0.9216 | 0.9792 | 0.9495 | 0.96731 |
| PCA 2 | KNN | 0.9732 | 0.9643 | 0.9696 | 0.0089 | 0.9216 | 0.9792 | 0.9495 | 0.96659 |
| PCA 2 | Logistic Regression | 0.9678 | 0.9643 | 0.9642 | 0.0035 | 0.9388 | 0.9583 | 0.9485 | 0.95938 |

*Note: The values above reproduce the supplied results table. The full experiment includes additional configurations.*

### Key Observations

- Random Forest with PCA 3 achieved the highest reported final score of `0.96778`.
- Its reported test accuracy was `97.14%`.
- PCA with only 2–3 components performed competitively in the supplied results.
- Random Forest achieved perfect training accuracy, while its test accuracy was lower, indicating a train-test performance gap.
- The final score is a custom ranking metric rather than a standard classification metric.

### WDBC Dataset

The notebook evaluates the WDBC dataset using the same four classifiers and four PCA configurations. Populate this table with the corresponding measured results from the notebook.

| PCA Configuration | Best Model | Test Accuracy | F1-score | ROC-AUC |
|---|---|---:|---:|---:|
| To be filled | To be filled | To be filled | To be filled | To be filled |

## Model Selection Strategy

The notebook ranks model configurations using a custom weighted score:

- F1-score: 40%
- Recall: 30%
- Cross-validation score: 20%
- Train-test accuracy gap component: 10%

This approach emphasizes classification performance, recall, validation performance, and the train-test accuracy gap.

## How to Run

### Prerequisites

Python 3.10 or later is recommended.

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn joblib
```

### Running the Notebook

1. Open the notebook in Google Colab.
2. Upload `wdbc.data`.
3. Upload `breast-cancer-wisconsin.data`.
4. Run the notebook cells in order.
5. Review the result tables, plots, and evaluation metrics.

The current notebook uses Google Colab's file-upload functionality, so the datasets must be uploaded when prompted.

## Limitations and Future Improvements

- Use an end-to-end Scikit-learn Pipeline to prevent preprocessing leakage during cross-validation.
- Fit imputation, scaling, and PCA separately within each training fold.
- Save the complete fitted preprocessing and prediction pipeline rather than only the classifier.
- Add automated experiment tracking and reproducibility checks.
- Explore additional classifiers and feature-selection methods.
- Report confidence intervals and repeated validation results.
- Document the final measured results for both datasets.



**Author:** Padmapriya R
