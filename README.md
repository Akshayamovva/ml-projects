Project 1 Global Temperature and GHG Emissions Analysis

This project explores the relationship between global temperature changes and greenhouse gas (GHG) emissions using a large-scale dataset of 8.6M+ records. The goal is to develop and evaluate predictive regression models that accurately estimate temperature variations based on emission data.

Project Overview
Objective: Study the correlation between global temperature rise and GHG emissions.
Dataset Size: 8,599,212 × 7

Data Preprocessing:
Mean imputation of missing values based on the past 5 years.
Feature scaling and transformation for model readiness.

Modeling Approach
Trained and tuned multiple regression models using Pipelines and GridSearchCV for hyperparameter optimization:
Linear Regression
Polynomial Regression
Ridge & Lasso Regression
Decision Tree Regressor
K-Nearest Neighbors (KNN)
Random Forest Regressor
XGBoost Regressor

Evaluation
All models were evaluated using R² Score, Mean Squared Error (MSE), and Root Mean Squared Error (RMSE) for accurate performance comparison.
Best Model: Random Forest Regressor R² Score: 0.873
Delivered the most reliable predictions with strong generalization.

Tech Stack
Languages & Libraries: Python, Pandas, NumPy, Scikit-Learn, XGBoost
Tools: Jupyter Notebook, Pipelines, GridSearchCV
PROJECT 2 Vehicle Insurance Claim Prediction

This project focuses on predicting whether a customer will claim vehicle insurance using machine learning techniques. The dataset contains 381,109 records and 12 features, and the objective is to build an accurate classification model to help insurance companies identify potential claimants.

 Project Overview

Goal: Predict the likelihood of a customer filing a vehicle insurance claim.

Dataset Shape: 381,109 × 12

Target Variable: Response (1 → Claim, 0 → No Claim)

 Data Preprocessing

Performed data cleaning and transformation to ensure high-quality inputs for modeling:

Outlier Removal: Identified and removed extreme values.

Missing Values: Imputed using mean for numerical and mode for categorical features.

Feature Encoding: Converted categorical variables into numeric format.

Data Balancing: Applied SMOTE (Synthetic Minority Oversampling Technique) to address class imbalance.

Exploratory Data Analysis (EDA)

Explored relationships between customer demographics, vehicle details, and claim probability.

Identified key patterns and trends influencing the likelihood of claims.

Visualized feature distributions and correlations using histograms, boxplots, and heatmaps.

 Machine Learning Models

Implemented and compared multiple classification algorithms:

K-Nearest Neighbors (KNN)

Logistic Regression

Naïve Bayes

Decision Tree

Random Forest

XGBoost Classifier

Feature Selection

Applied SelectKBest (Chi² test) to choose the most impactful features.

 Model Optimization

Used Pipelines, ColumnTransformer, and RandomizedSearchCV for hyperparameter tuning and workflow automation.

 Model Evaluation

Evaluated model performance using the following metrics:

Accuracy

Precision, Recall, F1-Score (Classification Report)

Best Model: Random Forest Classifier

Accuracy: 0.89

Delivered the best balance of precision and recall.

Tech Stack

Languages & Libraries: Python, Pandas, NumPy, Scikit-Learn, XGBoost, Imbalanced-learn

Tools: Jupyter Notebook, Matplotlib, Seaborn

Key Insights

Age, vehicle age, and driving experience were significant predictors of claim likelihood.

Ensemble methods like Random Forest provided robust and generalizable performance.

The project demonstrates the power of machine learning in risk assessment and customer targeting within the insurance domain.
