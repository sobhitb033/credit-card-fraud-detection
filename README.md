# Credit Card Fraud Detection using Machine Learning

A complete machine learning project for detecting fraudulent credit card transactions using supervised classification algorithms, SMOTE-based class balancing, and an interactive Streamlit web application.

The project compares Logistic Regression, Decision Tree, and Random Forest models and deploys the trained Random Forest model through a user-friendly web interface.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Project Workflow](#project-workflow)
- [Dataset](#dataset)
- [Dataset Statistics](#dataset-statistics)
- [Data Preprocessing](#data-preprocessing)
- [Class Imbalance](#class-imbalance)
- [SMOTE](#smote)
- [Train-Test Split](#train-test-split)
- [Machine Learning Models](#machine-learning-models)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Confusion Matrix Analysis](#confusion-matrix-analysis)
- [Final Model](#final-model)
- [Streamlit Application](#streamlit-application)
- [Application Features](#application-features)
- [Project Structure](#project-structure)
- [File Description](#file-description)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the Application](#running-the-application)
- [Running the Jupyter Notebook](#running-the-jupyter-notebook)
- [Using the Application](#using-the-application)
- [Reproducing the Results](#reproducing-the-results)
- [Documentation Guide](#documentation-guide)
- [Recommended Figures](#recommended-figures)
- [Recommended Tables](#recommended-tables)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Technologies Used](#technologies-used)
- [Troubleshooting](#troubleshooting)
- [Academic Project Summary](#academic-project-summary)
- [Authors](#authors)
- [License](#license)

---

# Project Overview

Credit card fraud is a major problem in digital financial transactions. Fraudulent transactions usually represent only a very small percentage of all transactions, making fraud detection a highly imbalanced binary classification problem.

This project develops a machine learning pipeline capable of classifying transactions as either:

- Legitimate
- Fraudulent

Three supervised machine learning algorithms are evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

SMOTE (Synthetic Minority Oversampling Technique) is used to address the severe class imbalance during model training.

After model evaluation, the trained Random Forest model is integrated into a Streamlit web application that allows users to:

- Enter transaction features manually
- Load demonstration transactions
- Obtain fraud/legitimate predictions
- View prediction probabilities
- Analyze transaction risk
- Explore dataset statistics
- Compare model performance
- Examine confusion matrices
- Understand the complete ML workflow

---

# Problem Statement

Credit card transaction datasets are highly imbalanced because fraudulent transactions occur much less frequently than legitimate transactions.

A conventional classifier can achieve very high accuracy simply by predicting most transactions as legitimate. Therefore, accuracy alone is not sufficient for evaluating a fraud detection system.

The objective of this project is to develop a classification system that can identify fraudulent transactions while maintaining a reasonable balance between:

- Precision
- Recall
- F1-score
- Accuracy

Particular attention is given to precision and recall because false positives and false negatives have different implications in fraud detection.

---

# Objectives

The major objectives of this project are:

1. Analyze a real-world credit card transaction dataset.
2. Identify and remove duplicate records.
3. Analyze the severe class imbalance between legitimate and fraudulent transactions.
4. Preprocess the transaction features.
5. Scale numerical features where required.
6. Divide the dataset into training and testing sets.
7. Apply SMOTE to balance the training data.
8. Train multiple machine learning classification models.
9. Evaluate the models using multiple performance metrics.
10. Compare the classification performance of the models.
11. Select the Random Forest model for deployment based on its measured performance.
12. Save the trained model and scaler for later inference.
13. Develop a Streamlit-based interactive application.
14. Provide an easy-to-use interface for fraud prediction.
15. Provide supporting visualizations and model analysis.
16. Document the complete methodology and experimental results.

---

# Project Workflow

The complete system follows the workflow below:

```text
                  ┌───────────────────────┐
                  │   Credit Card Dataset │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Remove Duplicates     │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Exploratory Analysis  │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Feature / Target      │
                  │ Separation            │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Feature Scaling       │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Train-Test Split      │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ SMOTE on Training Set │
                  └───────────┬───────────┘
                              │
                              ▼
       ┌────────────────────────────────────────────┐
       │           Model Training                   │
       │                                            │
       │  Logistic Regression                       │
       │  Decision Tree                             │
       │  Random Forest                             │
       └──────────────────────┬─────────────────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Model Evaluation      │
                  │                       │
                  │ Accuracy              │
                  │ Precision             │
                  │ Recall                │
                  │ F1-score              │
                  │ Confusion Matrix      │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Random Forest Model   │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Save Model + Scaler   │
                  └───────────┬───────────┘
                              │
                              ▼
                  ┌───────────────────────┐
                  │ Streamlit Application │
                  └───────────────────────┘
```

---

# Dataset

The project uses a standard credit card fraud detection dataset containing anonymized transaction features.

Each transaction contains 30 input features:

- `Time`
- `V1` to `V28`
- `Amount`

The target variable is:

- `Class`

where:

```text
Class = 0 → Legitimate transaction
Class = 1 → Fraudulent transaction
```

The `V1` to `V28` features are anonymized numerical features.

The `Time` feature represents the elapsed time from the first transaction in the dataset.

The `Amount` feature represents the transaction amount.

---

# Dataset Statistics

The original dataset contains:

```text
284,807 transactions
```

After duplicate removal, the cleaned dataset contains:

```text
283,726 transactions
```

Class distribution after duplicate removal:

| Class | Meaning | Count | Percentage |
|------:|---------|------:|-----------:|
| 0 | Legitimate | 283,253 | ~99.833% |
| 1 | Fraudulent | 473 | ~0.167% |
| **Total** | | **283,726** | **100%** |

This demonstrates the extreme class imbalance present in the dataset.

---

# Data Preprocessing

## 1. Duplicate Removal

Duplicate transactions are removed from the original dataset.

The dataset is reduced from:

```text
284,807 transactions
```

to:

```text
283,726 transactions
```

---

## 2. Feature and Target Separation

The target variable is:

```text
Class
```

The remaining transaction attributes are used as input features:

```text
Time, V1, V2, ..., V28, Amount
```

---

## 3. Feature Scaling

A scaler is used to transform the input features before model training.

The fitted scaler is saved as:

```text
models/fraud_scaler.pkl
```

The same saved scaler is used during inference in the Streamlit application.

Using the same preprocessing transformation during training and prediction is important because the model expects inputs in the same feature representation used during training.

---

# Class Imbalance

One of the major challenges in this dataset is the extreme difference between legitimate and fraudulent transactions.

After duplicate removal:

```text
Legitimate transactions: 283,253
Fraudulent transactions:     473
```

Fraudulent transactions therefore represent only approximately:

```text
0.167%
```

of the cleaned dataset.

A model trained directly on this distribution may become biased toward the majority class.

For this reason, SMOTE is applied to the training data.

---

# SMOTE

SMOTE stands for:

**Synthetic Minority Oversampling Technique**

Instead of simply duplicating minority-class samples, SMOTE generates synthetic samples based on existing minority-class observations.

SMOTE is applied **only to the training dataset**.

This is important because applying SMOTE before splitting the dataset could introduce information leakage between the training and testing sets.

The training data before SMOTE contains approximately:

```text
Legitimate: 226,602
Fraudulent:     378
```

After applying SMOTE:

```text
Legitimate: 226,602
Fraudulent: 226,602
```

The test dataset is kept separate and retains its original class distribution.

---

# Train-Test Split

The cleaned dataset is divided into:

- Approximately 80% training data
- Approximately 20% testing data

The training set is used to:

- Fit the scaler
- Apply SMOTE
- Train the machine learning models

The test set is kept separate for final model evaluation.

This separation helps provide a more realistic estimate of model performance on unseen transactions.

---

# Machine Learning Models

Three classification algorithms are evaluated.

---

## 1. Logistic Regression

Logistic Regression is a linear classification algorithm used for binary classification.

It estimates the probability that a transaction belongs to the fraudulent class.

The logistic function is:

```text
σ(z) = 1 / (1 + e^(-z))
```

Logistic Regression provides a useful baseline model for comparison.

---

## 2. Decision Tree

A Decision Tree recursively divides the dataset using feature-based decision rules.

Each internal node represents a decision condition, while leaf nodes represent predicted classes.

Decision Trees can capture nonlinear relationships between features and the target.

However, individual decision trees can be sensitive to training data and may overfit.

---

## 3. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees.

Each tree produces a prediction and the ensemble combines these predictions to produce the final classification.

Random Forest can model nonlinear relationships and interactions between features.

The trained Random Forest model is saved as:

```text
models/random_forest_fraud_model.pkl
```

---

# Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Because the dataset is highly imbalanced, accuracy should not be considered independently.

---

## Accuracy

Accuracy measures the proportion of correctly classified transactions.

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

---

## Precision

Precision measures how many transactions predicted as fraudulent were actually fraudulent.

```text
Precision = TP / (TP + FP)
```

High precision means fewer legitimate transactions are incorrectly flagged as fraud.

---

## Recall

Recall measures how many actual fraudulent transactions were successfully detected.

```text
Recall = TP / (TP + FN)
```

High recall means fewer fraudulent transactions are missed.

---

## F1-Score

F1-score is the harmonic mean of precision and recall.

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

F1-score is particularly useful for imbalanced classification problems because it considers both false positives and false negatives.

---

# Results

The three evaluated models produced the following results:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.37% | 5.30% | 87.37% | 10.00% |
| Decision Tree | 99.74% | 35.26% | 64.21% | 45.52% |
| Random Forest | **99.95%** | **91.25%** | 76.84% | **83.43%** |

Random Forest achieved the highest measured accuracy, precision, and F1-score among the evaluated models.

Logistic Regression achieved the highest recall among the three evaluated models, but its precision was substantially lower.

This illustrates why multiple evaluation metrics are necessary for fraud detection.

---

# Confusion Matrix Analysis

A confusion matrix contains four categories:

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | True Negative (TN) | False Positive (FP) |
| Actual Fraud | False Negative (FN) | True Positive (TP) |

---

## Logistic Regression

The evaluated Logistic Regression model produced:

```text
TN = 55,169
FP = 1,482
FN = 12
TP = 83
```

The model detected a large proportion of fraudulent transactions, but generated a relatively high number of false positives.

---

## Random Forest

The evaluated Random Forest model produced:

```text
TN = 56,644
FP = 7
FN = 22
TP = 73
```

The Random Forest model produced substantially fewer false positives while maintaining a high fraud-detection capability.

---

# Final Model

The Random Forest model is used for deployment based on its measured performance among the evaluated models.

The trained model is stored at:

```text
models/random_forest_fraud_model.pkl
```

The corresponding scaler is stored at:

```text
models/fraud_scaler.pkl
```

During prediction, the application:

1. Receives transaction feature values.
2. Arranges them in the expected feature order.
3. Applies the saved scaler.
4. Passes the scaled data to the Random Forest model.
5. Obtains the predicted class.
6. Obtains prediction probabilities.
7. Displays the result through the Streamlit interface.

Expected feature order:

```text
Time
V1
V2
V3
V4
V5
V6
V7
V8
V9
V10
V11
V12
V13
V14
V15
V16
V17
V18
V19
V20
V21
V22
V23
V24
V25
V26
V27
V28
Amount
```

---

# Streamlit Application

The machine learning model is deployed through a Streamlit web application.

The application entry point is:

```text
app.py
```

The application provides an interactive interface for performing fraud detection without requiring users to directly interact with Python code.

---

# Application Features

## 1. Quick Demo

The application provides demonstration transactions that can be loaded into the input fields.

Two types of examples are provided:

- Legitimate transaction
- Fraudulent transaction

This allows users to quickly test the application.

---

## 2. Manual Transaction Input

Users can manually enter values for:

```text
Time
V1 - V28
Amount
```

The application then performs a prediction.

---

## 3. Prediction

The application displays the predicted transaction class as either:

```text
Legitimate Transaction
```

or:

```text
Fraudulent Transaction
```

---

## 4. Prediction Probability

The application displays estimated probabilities for:

- Legitimate transaction
- Fraudulent transaction

Example:

```text
Legitimate Probability: 45%
Fraud Probability: 55%
```

---

## 5. Risk Analysis

Based on the model output, the application provides a risk interpretation.

Possible risk levels include:

- Low Risk
- Medium Risk
- High Risk

The risk interpretation is intended for demonstration and educational purposes.

---

## 6. Dataset Overview

The application provides an overview of the cleaned dataset, including:

- Total transactions
- Legitimate transactions
- Fraudulent transactions
- Fraud percentage

---

## 7. Class Distribution

The application visualizes the distribution between:

- Legitimate transactions
- Fraudulent transactions

This demonstrates the severe class imbalance in the dataset.

---

## 8. Model Comparison

The application displays the performance of:

- Logistic Regression
- Decision Tree
- Random Forest

using:

- Accuracy
- Precision
- Recall
- F1-score

---

## 9. Confusion Matrix Analysis

The application provides confusion matrix information to help users understand:

- True positives
- True negatives
- False positives
- False negatives

---

## 10. System Explanation

The application contains a section explaining:

- Dataset
- Preprocessing
- SMOTE
- Model training
- Evaluation
- Prediction

This makes the application useful for project demonstrations and academic presentations.

---

# Project Structure

```text
credit-card-fraud-detection/
│
├── data/
│   └── creditcard.csv
│
├── models/
│   ├── fraud_scaler.pkl
│   └── random_forest_fraud_model.pkl
│
├── notebooks/
│   └── fraud_detection.ipynb
│
├── src/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

---

# File Description

## `app.py`

Main Streamlit application.

Responsible for:

- Loading the trained model
- Loading the scaler
- Loading dataset information
- Accepting user input
- Generating predictions
- Displaying probabilities
- Displaying risk analysis
- Displaying model evaluation information

---

## `data/creditcard.csv`

The credit card transaction dataset.

The file is approximately 144 MB and is stored using **Git LFS**.

---

## `models/fraud_scaler.pkl`

Saved preprocessing scaler.

Used to transform input features before prediction.

---

## `models/random_forest_fraud_model.pkl`

Trained Random Forest classifier used by the Streamlit application.

---

## `notebooks/fraud_detection.ipynb`

Complete machine learning development notebook.

It contains:

- Data loading
- Data inspection
- Duplicate removal
- Exploratory analysis
- Class distribution analysis
- Feature preprocessing
- Train-test split
- SMOTE
- Model training
- Model prediction
- Evaluation
- Confusion matrices
- Model comparison
- Model saving

---

## `requirements.txt`

Contains the Python packages required to run the project.

---

## `.gitignore`

Prevents unnecessary files from being uploaded to GitHub, including:

- Virtual environments
- Python cache
- Jupyter checkpoints
- IDE files
- macOS system files
- Environment/secrets files

---

## `.gitattributes`

Configures Git LFS for the large dataset.

The dataset is tracked using Git LFS:

```text
data/creditcard.csv
```

---

# Requirements

Before running the project, the system should have:

- Python 3.9 or later
- Git
- Git LFS
- pip
- Internet connection for installing dependencies

The project can run on:

- macOS
- Windows
- Linux

---

# Installation

## Step 1 — Install Git

Check whether Git is installed:

```bash
git --version
```

---

## Step 2 — Install Git LFS

Git LFS is required because the dataset is approximately 144 MB.

Check:

```bash
git lfs --version
```

If Git LFS is not installed:

### macOS

Using Homebrew:

```bash
brew install git-lfs
```

Then:

```bash
git lfs install
```

### Windows

Install Git LFS and then run:

```bash
git lfs install
```

### Linux

Install Git LFS using the package manager appropriate for your Linux distribution, then run:

```bash
git lfs install
```

---

# Clone the Repository

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git
```

Replace `YOUR_USERNAME` with the GitHub account that owns this repository.

Navigate into the project:

```bash
cd credit-card-fraud-detection
```

---

# Download the Git LFS Dataset

Because the dataset is stored using Git LFS, verify the LFS files:

```bash
git lfs ls-files
```

You should see:

```text
data/creditcard.csv
```

If the dataset has not been downloaded correctly, run:

```bash
git lfs pull
```

Verify its size:

```bash
ls -lh data/creditcard.csv
```

The file should be approximately 144 MB.

---

# Create a Virtual Environment

It is recommended to use a virtual environment.

Run:

```bash
python3 -m venv venv
```

---

# Activate the Virtual Environment

## macOS / Linux

```bash
source venv/bin/activate
```

## Windows

```bash
venv\Scripts\activate
```

After activation, the terminal should show:

```text
(venv)
```

---

# Upgrade pip

Run:

```bash
python -m pip install --upgrade pip
```

---

# Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

This installs all required packages.

---

# Running the Application

After activating the virtual environment and installing the dependencies:

```bash
streamlit run app.py
```

Streamlit will start a local server.

The terminal will normally display:

```text
Local URL: http://localhost:8501
```

Open the displayed URL in a browser.

---

# Running the Jupyter Notebook

The complete machine learning workflow is available in:

```text
notebooks/fraud_detection.ipynb
```

If Jupyter is not installed:

```bash
pip install notebook
```

Then run:

```bash
jupyter notebook
```

Open:

```text
notebooks/fraud_detection.ipynb
```

Run the notebook cells sequentially.

---

# Using the Application

After running:

```bash
streamlit run app.py
```

the application provides multiple sections.

## Quick Demonstration

Use the demonstration controls to load a sample transaction.

The application will populate the transaction fields.

Then run the prediction.

---

## Manual Prediction

Enter values for:

```text
Time
V1
V2
V3
...
V28
Amount
```

Then click the prediction button.

The application will display:

- Prediction
- Legitimate probability
- Fraud probability
- Risk level

---

# Reproducing the Machine Learning Results

The complete experimental workflow can be reproduced using:

```text
notebooks/fraud_detection.ipynb
```

The notebook follows this sequence:

```text
Load Dataset
      ↓
Inspect Dataset
      ↓
Remove Duplicates
      ↓
Analyze Class Distribution
      ↓
Separate Features and Target
      ↓
Train-Test Split
      ↓
Feature Scaling
      ↓
Apply SMOTE to Training Data
      ↓
Train Logistic Regression
      ↓
Train Decision Tree
      ↓
Train Random Forest
      ↓
Generate Predictions
      ↓
Calculate Metrics
      ↓
Generate Confusion Matrices
      ↓
Compare Models
      ↓
Save Random Forest Model
      ↓
Save Scaler
```

---

# Important Reproducibility Notes

## SMOTE

SMOTE is applied only to the training data.

The test data is not oversampled.

This prevents synthetic samples from appearing in the evaluation set and helps maintain a more realistic evaluation.

## Scaling

The scaler is fitted during preprocessing and saved.

The same scaler is reused by the Streamlit application during prediction.

## Model

The trained Random Forest model is saved as:

```text
models/random_forest_fraud_model.pkl
```

---

# Documentation Guide

This section provides the information required to prepare the academic project documentation or IEEE-format paper.

---

## 1. Abstract

The abstract should cover:

- The problem of credit card fraud
- Highly imbalanced transaction data
- Data preprocessing
- SMOTE
- Machine learning models
- Model comparison
- Random Forest
- Final performance
- Streamlit deployment

Important measured Random Forest results:

```text
Accuracy  = 99.95%
Precision = 91.25%
Recall    = 76.84%
F1-Score  = 83.43%
```

---

## 2. Introduction

The introduction should discuss:

- Growth of digital payments
- Increase in electronic financial transactions
- Importance of fraud detection
- Financial and operational impact of fraud
- Difficulty of detecting rare fraudulent transactions
- Class imbalance
- Machine learning as a solution
- Motivation for the proposed system

---

## 3. Problem Statement

Explain that fraudulent transactions form a very small minority of all transactions.

Therefore, a classification system must detect fraud while controlling:

- False positives
- False negatives

Accuracy alone is insufficient because of the severe class imbalance.

---

## 4. Objectives

The documentation should include:

1. Data cleaning
2. Duplicate removal
3. Exploratory data analysis
4. Feature preprocessing
5. Class imbalance analysis
6. SMOTE
7. Train-test split
8. Model training
9. Model evaluation
10. Model comparison
11. Random Forest deployment
12. Streamlit application development

---

## 5. Related Work

The related work section can discuss:

- Traditional rule-based fraud detection
- Logistic Regression
- Decision Trees
- Random Forest
- Ensemble learning
- Imbalanced classification
- SMOTE
- Machine learning based fraud detection

Academic references should be added to the final paper.

---

## 6. Dataset Description

Include:

```text
Original transactions: 284,807
Cleaned transactions: 283,726
Legitimate transactions: 283,253
Fraudulent transactions: 473
Fraud percentage: ~0.167%
```

Features:

```text
Time
V1
V2
...
V28
Amount
```

Target:

```text
Class
```

Class meanings:

```text
0 → Legitimate
1 → Fraudulent
```

---

## 7. Data Preprocessing

Explain:

1. Dataset loading
2. Duplicate removal
3. Feature-target separation
4. Train-test split
5. Feature scaling
6. SMOTE applied only to training data

---

## 8. Proposed Methodology

The methodology should explain:

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Processing
   ↓
Train-Test Split
   ↓
SMOTE
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Random Forest Selection
   ↓
Model Serialization
   ↓
Streamlit Deployment
```

---

## 9. Algorithms

Create separate subsections for:

### Logistic Regression

Explain:

- Binary classification
- Logistic function
- Probability estimation
- Baseline comparison

### Decision Tree

Explain:

- Recursive splitting
- Decision nodes
- Leaf nodes
- Nonlinear relationships

### Random Forest

Explain:

- Ensemble learning
- Multiple decision trees
- Aggregation of predictions
- Ability to capture nonlinear relationships

---

## 10. Evaluation Metrics

Explain:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Include mathematical equations.

### Accuracy

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

### Precision

```text
Precision = TP / (TP + FP)
```

### Recall

```text
Recall = TP / (TP + FN)
```

### F1-Score

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

---

## 11. Experimental Results

Use this table:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.37% | 5.30% | 87.37% | 10.00% |
| Decision Tree | 99.74% | 35.26% | 64.21% | 45.52% |
| Random Forest | 99.95% | 91.25% | 76.84% | 83.43% |

Discuss why accuracy alone is not sufficient.

---

## 12. Confusion Matrix Analysis

### Logistic Regression

```text
TN = 55,169
FP = 1,482
FN = 12
TP = 83
```

### Random Forest

```text
TN = 56,644
FP = 7
FN = 22
TP = 73
```

Explain:

- TN = Legitimate transactions correctly identified
- FP = Legitimate transactions incorrectly classified as fraud
- FN = Fraudulent transactions incorrectly classified as legitimate
- TP = Fraudulent transactions correctly identified

---

## 13. System Implementation

Explain the Streamlit implementation.

Include:

- Model loading
- Scaler loading
- Input handling
- Feature ordering
- Feature scaling
- Prediction
- Probability calculation
- Risk analysis
- Dataset statistics
- Model comparison
- Confusion matrix analysis

---

## 14. System Architecture

A suitable architecture is:

```text
                    USER
                      │
                      ▼
             ┌─────────────────┐
             │  Streamlit UI   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Input Validation│
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Saved Scaler    │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Random Forest   │
             │ Model           │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Prediction +    │
             │ Probabilities   │
             └────────┬────────┘
                      │
                      ▼
             ┌─────────────────┐
             │ Risk Analysis   │
             └─────────────────┘
```

---

# Recommended Figures

For a complete academic report, include the following figures where applicable.

## Figure 1 — Dataset Class Distribution

Show:

```text
Legitimate Transactions vs Fraudulent Transactions
```

This demonstrates the severe class imbalance.

---

## Figure 2 — Project Workflow

Show:

```text
Dataset
→ Cleaning
→ Preprocessing
→ SMOTE
→ Model Training
→ Evaluation
→ Deployment
```

---

## Figure 3 — Model Comparison

Create a bar chart comparing:

- Accuracy
- Precision
- Recall
- F1-score

for:

- Logistic Regression
- Decision Tree
- Random Forest

---

## Figure 4 — Logistic Regression Confusion Matrix

Use:

```text
TN = 55,169
FP = 1,482
FN = 12
TP = 83
```

---

## Figure 5 — Random Forest Confusion Matrix

Use:

```text
TN = 56,644
FP = 7
FN = 22
TP = 73
```

---

## Figure 6 — Streamlit Application Interface

Take a screenshot showing the main application.

---

## Figure 7 — Legitimate Prediction

Take a screenshot showing the application predicting a legitimate transaction.

---

## Figure 8 — Fraud Prediction

Take a screenshot showing the application predicting a fraudulent transaction.

---

# Recommended Tables

## Table 1 — Dataset Distribution

| Class | Count | Percentage |
|---|---:|---:|
| Legitimate | 283,253 | ~99.833% |
| Fraudulent | 473 | ~0.167% |

---

## Table 2 — Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 97.37% | 5.30% | 87.37% | 10.00% |
| Decision Tree | 99.74% | 35.26% | 64.21% | 45.52% |
| Random Forest | 99.95% | 91.25% | 76.84% | 83.43% |

---

## Table 3 — Random Forest Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 56,644 | 7 |
| Actual Fraud | 22 | 73 |

---

## Table 4 — Logistic Regression Confusion Matrix

| | Predicted Legitimate | Predicted Fraud |
|---|---:|---:|
| Actual Legitimate | 55,169 | 1,482 |
| Actual Fraud | 12 | 83 |

---

# Limitations

The project has several limitations.

## 1. Historical Dataset

The model is trained using a historical dataset and may not represent current fraud patterns.

## 2. Anonymized Features

Most transaction features are anonymized, which limits direct interpretation of individual variables.

## 3. Class Imbalance

Although SMOTE is used during training, the real-world distribution remains highly imbalanced.

## 4. Static Model

The deployed model does not automatically retrain when new transaction data becomes available.

## 5. Demonstration Application

The Streamlit application is intended for academic and demonstration purposes and should not be considered a production banking fraud detection system.

## 6. Probability Interpretation

The model's predicted probabilities should not automatically be interpreted as calibrated real-world probabilities without additional calibration analysis.

---

# Future Scope

Potential improvements include:

1. Real-time transaction stream processing
2. Continuous model retraining
3. Online learning
4. Hyperparameter optimization
5. Threshold optimization based on business costs
6. Probability calibration
7. Explainable AI using SHAP or similar techniques
8. Advanced anomaly detection
9. Gradient boosting models such as XGBoost or LightGBM
10. Deep learning approaches
11. Real-time dashboards
12. Cloud deployment
13. REST API integration
14. Database-backed transaction monitoring
15. Alert and notification systems
16. Model drift detection
17. Automated model retraining
18. Production-grade authentication and authorization

---

# Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical computation |
| Scikit-learn | Machine learning |
| Imbalanced-learn | SMOTE |
| Matplotlib | Data visualization |
| Seaborn | Data visualization |
| Joblib | Model serialization |
| Jupyter Notebook | Model development |
| Streamlit | Web application |
| Git | Version control |
| Git LFS | Large dataset storage |
| GitHub | Source code hosting |

---

# Complete Installation Commands

For someone setting up the project from scratch:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git

cd credit-card-fraud-detection

git lfs install

git lfs pull

python3 -m venv venv

source venv/bin/activate

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit run app.py
```

### Windows

Use:

```bash
git clone https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git

cd credit-card-fraud-detection

git lfs install

git lfs pull

python -m venv venv

venv\Scripts\activate

python -m pip install --upgrade pip

pip install -r requirements.txt

streamlit run app.py
```

---

# Troubleshooting

## Dataset is only a few KB

If:

```bash
ls -lh data/creditcard.csv
```

shows a very small file instead of approximately 144 MB, run:

```bash
git lfs pull
```

Then verify:

```bash
git lfs ls-files
```

---

## `streamlit: command not found`

Make sure the virtual environment is activated:

```bash
source venv/bin/activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## `ModuleNotFoundError`

Run:

```bash
pip install -r requirements.txt
```

---

## Model file not found

Make sure these files exist:

```text
models/fraud_scaler.pkl
models/random_forest_fraud_model.pkl
```

---

## Python version problems

Check:

```bash
python --version
```

A modern Python 3 version is recommended.

---

## Port 8501 is already in use

Run Streamlit on another port:

```bash
streamlit run app.py --server.port 8502
```

---

# Academic Project Summary

This project demonstrates the application of machine learning to highly imbalanced credit card fraud detection.

The study evaluates Logistic Regression, Decision Tree, and Random Forest classifiers after applying appropriate preprocessing and SMOTE-based balancing to the training data.

Among the evaluated models, Random Forest achieved:

```text
Accuracy  = 99.95%
Precision = 91.25%
Recall    = 76.84%
F1-Score  = 83.43%
```

The trained Random Forest model is subsequently integrated into a Streamlit application that provides interactive fraud prediction and visualization capabilities.

The project demonstrates the complete machine learning lifecycle:

```text
Data Collection
      ↓
Data Cleaning
      ↓
Exploratory Data Analysis
      ↓
Preprocessing
      ↓
Class Balancing
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Model Selection
      ↓
Model Serialization
      ↓
Application Development
      ↓
Deployment / Demonstration
```

---
