# 📌 Contract Renewal Prediction – Quick Notes

## 🎯 Goal

Predict whether a customer will renew a contract and help decide discount strategy.

---

## 📊 Steps in Pipeline

### 1. Load Data

* Use pandas to read CSV

### 2. Data Cleaning

* Handle missing values using fillna()

### 3. Feature Engineering

* Create new features like:

  * Discount_Per_Contract

---

### 4. Feature Selection

* Remove:

  * Customer_ID
  * Serial_No
* Target:

  * Renewal

---

### 5. Train-Test Split

* 80% training
* 20% testing

---

### 6. Model Selection

* Decision Tree
* Random Forest
* XGBoost (Best)

---

### 7. Cross Validation

* Use 5-fold CV
* Check model stability

---

### 8. Model Training

* Fit model on training data

---

### 9. Prediction

* predict() → class (0/1)
* predict_proba() → probability

---

### 10. Evaluation

* Accuracy
* Precision
* Recall (Important)
* F1-score

---

### 11. Save Model

* Use pickle

---

### 12. Deployment

* Use Streamlit
* Take user input
* Predict renewal probability

---

## 🔥 Key Learnings

* Don’t trust 100% accuracy (overfitting)
* Always check recall for business problems
* Feature engineering improves model
* Probability is more useful than prediction

---

## 🚀 Final Output

* Predict renewal
* Suggest discount based on probability

---
