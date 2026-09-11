# Car-Insurance-Fraud-Detection-Using-Machine-Learning
Auto Insurance Fraud Detection System

A machine learning pipeline that detects fraudulent auto insurance claims using supervised classification models, feature selection, and exploratory data analysis.

📌 Overview

Insurance fraud costs providers billions annually, driving up premiums for genuine policyholders and eroding trust in the industry. This project builds an end-to-end fraud detection system that classifies auto insurance claims as genuine or fraudulent based on policy and claim attributes, using Python-based machine learning.

🎯 Objective

Identify patterns that distinguish fraudulent claims from legitimate ones.
Build and compare multiple classification models to determine the most effective approach.
Reduce reliance on manual claim review by automating fraud detection.

📊 Dataset

Source: Kaggle – Vehicle Claim Fraud Detection
Size: 15,420 records × 33 features (24 categorical, 9 numerical)
Target variable: FraudFound_P (0 = Genuine, 1 = Fraudulent)
Class distribution: 94% genuine claims vs. 6% fraudulent — a highly imbalanced dataset

🛠️ Tools & Technologies

Language: Python
Libraries: scikit-learn, TensorFlow/Keras, Pandas, NumPy, Seaborn, Matplotlib, mlxtend
Environment: Jupyter Notebook

🔍 Methodology

Data Preprocessing — Checked for missing values, applied label encoding to categorical features.
Exploratory Data Analysis (EDA) — Analyzed class imbalance, demographic patterns (e.g., fraud by sex), and feature correlations via heatmaps.
Feature Selection — Used an Exhaustive Feature Selector (EFS) with a Decision Tree base estimator to reduce 33 features down to the 5 most predictive: Fault, PolicyType, AgeOfPolicyHolder, AddressChange_Claim, NumberOfCars.
Model Training — Trained and evaluated four classifiers on an 80/20 train-test split:
Random Forest Classifier
Gradient Boosting Classifier
Artificial Neural Network (Keras/TensorFlow)
Gaussian Naive Bayes
Evaluation — Assessed models using accuracy, precision, recall, F1-score, and confusion matrices (critical given the class imbalance).

📈 Results

Model	Accuracy	Precision (Fraud)	Recall (Fraud)	F1-Score (Fraud)
Random Forest Classifier	93.94%	0.86	0.06	0.11
Gradient Boosting Classifier	93.87%	0.90	0.05	0.09
Artificial Neural Network	93.84%	–	–	–
Gaussian Naive Bayes	91.21%	0.11	0.05	0.07

Random Forest Classifier emerged as the top-performing model. Despite high overall accuracy across all models, recall on the minority (fraud) class remained low — a direct consequence of severe class imbalance, highlighting an area for future improvement (e.g., SMOTE, class weighting).
