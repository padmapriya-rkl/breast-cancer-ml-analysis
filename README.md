
# Breast Cancer Classification & PCA

A machine learning project that compares classification models and
Principal Component Analysis (PCA) configurations on two Wisconsin
Breast Cancer datasets.

The project explores data preprocessing, feature scaling, dimensionality
reduction, hyperparameter tuning, cross-validation, and model evaluation.

## Project Overview

The objective is to compare machine learning models with different
PCA configurations and evaluate their classification performance
using multiple metrics.

Two datasets are studied:
- Wisconsin Diagnostic Breast Cancer (WDBC)
- Original Wisconsin Breast Cancer Dataset

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Google Colab

## Datasets

### 1. Wisconsin Diagnostic Breast Cancer (WDBC)
- 569 samples
- 30 numerical features before removing the ID column
- Binary diagnosis: Benign or Malignant

### 2. Original Wisconsin Breast Cancer Dataset
- 699 samples
- 9 predictive features
- Contains missing values represented by `?`
- Class labels are mapped to benign and malignant categories

Dataset source: UCI Machine Learning Repository.

## Project Workflow

1. Load and inspect both datasets.
2. Analyze missing values and class distributions.
3. Visualize feature distributions, correlations, and potential outliers.
4. Preprocess the data and separate features from target labels.
5. Split data into stratified training and testing sets.
6. Standardize features using training data.
7. Apply PCA with different numbers of components.
8. Train and compare classification models.
9. Tune selected model hyperparameters using GridSearchCV.
10. Evaluate models using cross-validation and test-set metrics.
11. Visualize confusion matrices, ROC curves, and learning curves.
12. Compare results across both datasets.

## Machine Learning Models

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- K-Nearest Neighbors (KNN)

## PCA Experiments

### WDBC Dataset
- No PCA: 30 features
- PCA: 15 components
- PCA: 10 components
- PCA: 5 components

### Original Wisconsin Dataset
- No PCA: 9 features
- PCA: 5 components
- PCA: 3 components
- PCA: 2 components

These configurations help investigate the trade-off between
dimensionality reduction and classification performance.

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Cross-validation score
- Train-test accuracy gap

The notebook also generates confusion matrices, ROC curves,
cumulative explained variance plots, and learning curves.

## Hyperparameter Tuning

GridSearchCV is used with stratified 5-fold cross-validation
for selected models.

The search covers parameters such as:
- SVM: C and gamma
- Random Forest: number of estimators and maximum depth
- KNN: number of neighbors

## Model Selection

A combined score ranks model configurations using:
- F1-score: 40%
- Recall: 30%
- Cross-validation score: 20%
- Overfitting-gap component: 10%

This scoring method emphasizes classification quality, recall,
validation performance, and the train-test accuracy gap.

## Results

The notebook generates result tables for each dataset and compares
the selected models using test accuracy and F1-score.

Add the final measured results here after running the notebook:

| Dataset | Best Model | PCA Configuration | Test Accuracy | F1-score |
|---|---|---|---|---|
| WDBC | To be filled | To be filled | To be filled | To be filled |
| Original Wisconsin | To be filled | To be filled | To be filled | To be filled |

## How to Run

1. Open the notebook in Google Colab.
2. Upload `wdbc.data`.
3. Upload `breast-cancer-wisconsin.data`.
4. Run the notebook cells in order.
5. Review the generated metrics, plots, and comparison tables.

## Future Improvements

- Use an end-to-end Scikit-learn Pipeline to prevent data leakage
  during cross-validation and PCA experiments.
- Add automated experiment tracking.
- Save the complete preprocessing and prediction pipeline.
- Evaluate additional classifiers and feature-selection techniques.
- Improve reproducibility with documented dependencies and tests.

## Disclaimer

This project is for educational and research purposes only.
It is not intended to diagnose breast cancer or replace professional
medical evaluation.
