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

