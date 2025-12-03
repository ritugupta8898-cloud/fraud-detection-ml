# 🛡️ Credit Card Fraud Detection — Machine Learning Project

This project builds a complete Machine Learning pipeline to detect fraudulent credit card transactions using a real-world dataset. It includes data preprocessing, oversampling, model training, evaluation, and visualization.

---

## 📌 Project Overview

Credit card fraud is rare (0.17% cases), making this an **imbalanced classification** problem.  
Accuracy is not enough, so we focus on:

- **Recall** (catching fraud)
- **Precision** (avoiding false alarms)
- **ROC-AUC** (model separation power)

---

## 🚀 Features Implemented

- Load and preprocess dataset  
- Scale `Time` and `Amount`  
- Handle imbalance with RandomOverSampler  
- Train Random Forest  
- Evaluate with accuracy, precision, recall, F1, ROC-AUC  
- Clean project structure (industry standard)

---

## 📁 Project Structure
fraud-detection-ml/
│── data/
│── src/
│ ├── load_data.py
│ ├── preprocess.py
│ ├── train.py
│ ├── evaluate.py
│── run.py
│── requirements.txt
│── .gitignore
│── README.md

---

## 📊 Model Performance (Random Forest)

| Metric      | Score |
|-------------|--------|
| Accuracy    | 0.9992 |
| Precision   | 0.74   |
| Recall      | 0.83   |
| F1-Score    | 0.78   |
| ROC-AUC     | 0.918  |

---

## 📊 Model Performance (Random Forest)

| Metric      | Score |
|-------------|--------|
| Accuracy    | 0.9992 |
| Precision   | 0.74   |
| Recall      | 0.83   |
| F1-Score    | 0.78   |
| ROC-AUC     | 0.918  |



## 🛠️ How to Run

pip install -r requirements.txt
python3 run.py

Kaggle: Credit Card Fraud Detection
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Ritu Gupta (Pratyush Gupta)
GitHub: https://github.com/rituGupta8898-cloud