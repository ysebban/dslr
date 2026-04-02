# DSLR – Data Science Logistic Regression

## 📌 Overview

DSLR is a data science project where the goal is to recreate a basic machine learning pipeline from scratch, without using high-level libraries such as scikit-learn.

The objective is to analyze a dataset of Hogwarts students and predict their house using a logistic regression model.

---

## 🧠 Project Goals

- Understand how data analysis works from raw CSV
- Implement statistical functions manually
- Visualize data distributions and correlations
- Build a logistic regression model from scratch
- Evaluate predictions on unseen data

---

## 📊 Dataset

The dataset contains various academic scores of Hogwarts students across different subjects.

Example features:
- Astronomy
- Herbology
- Defense Against the Dark Arts
- Potions
- Charms
- etc.

Target:
- Hogwarts House (classification)

---

## ⚙️ Features

### 1. Data Analysis
- Custom implementation of:
  - mean
  - standard deviation
  - quartiles
  - variance
- Handling missing values
- Feature selection based on variance / separation

### 2. Data Visualization
- Histograms per feature
- Scatter plots to analyze correlations
- Interactive navigation between plots

### 3. Machine Learning
- Logistic Regression implemented from scratch
- Gradient descent optimization
- One-vs-All classification

### 4. Prediction
- Model trained on dataset_train.csv
- Predictions generated on dataset_test.csv

---

## 🛠️ Tech Stack

- Python
- NumPy (optional depending on your version)
- Matplotlib
- CSV parsing (manual or pandas depending on constraints)

---

## 🚀 Usage

### Train the model
```bash
python3 logreg_train.py dataset_train.csv
