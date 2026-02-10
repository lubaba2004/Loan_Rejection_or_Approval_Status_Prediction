
# 🏦 Loan Rejection or Approval Status Prediction

## 📌 Project Overview

This project focuses on predicting whether a loan application will be **Approved** or **Rejected** using **Machine Learning classification models**.
It helps financial institutions assess loan risk based on applicant financial, personal, and asset information.

The project also includes an **interactive Streamlit web application** that allows users to input applicant details and get real-time loan approval predictions.

---

## 🎯 Problem Statement

Loan approval is a critical decision for banks and financial institutions.
Manual evaluation is time-consuming and may lead to biased decisions.

**Objective:**
To build a machine learning model that accurately predicts loan approval status based on applicant data and deploy it as a web application.

---

## 📂 Dataset Description

The dataset contains applicant information such as:

* Number of dependents
* Annual income
* Loan amount
* Loan term
* CIBIL score
* Residential, commercial, luxury, and bank asset values
* Education status
* Self-employment status

**Target Variable:**

* `loan_status`

  * `1` → Loan Approved
  * `0` → Loan Rejected

---

## 🛠️ Technologies Used

### 🔹 Programming & Tools

* Python
* Jupyter Notebook
* Git & GitHub

### 🔹 Libraries

* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

---

## 🤖 Machine Learning Models Used

The following classification models were trained and evaluated:

* Logistic Regression
* Decision Tree Classifier
* Random Forest Classifier
* Gradient Boosting Classifier
* K-Nearest Neighbors
* Gaussian Naive Bayes

---

## 📊 Model Evaluation Metrics

Models were evaluated using multiple metrics to ensure reliability:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC Score
* Confusion Matrix

The **best performing model** was selected based on **Test F1-Score and ROC-AUC**.


## 🚀 Streamlit Web Application

### Features:

* User-friendly input form
* Real-time loan approval prediction
* Prediction probability (confidence score)
* Clean and professional UI

### Sample Inputs:

* Income
* Loan amount
* Loan term
* CIBIL score
* Asset values
* Education & employment status

## 📁 Project Structure

Loan_Rejection_or_Approval_Status_Prediction/
│
├── app.py                  # Streamlit app
├── loan_model.pkl          # Trained ML model
├── requirements.txt        # Required libraries
├── README.md               # Project documentation
├── .gitignore              # Ignored files


## ✅ Key Outcomes

* Built an end-to-end ML classification pipeline
* Compared multiple ML models
* Selected best model using robust metrics
* Deployed a real-time prediction web app
* Improved understanding of model evaluation and deployment


## 📌 Future Improvements

* Add feature importance visualization
* Improve model performance with hyperparameter tuning
* Add authentication for secure access
* Integrate database for storing predictions

## 👩‍💻 Author

**Lubaba N**

# Loan Approval Prediction App 💳

A machine learning–based Streamlit application that predicts whether a loan will be **approved or rejected** based on applicant details.

## 🚀 Live Demo
👉 https://loan-rejection-or-approval-status-prediction.streamlit.app

## 📊 Features
- User-friendly input form
- Real-time loan approval prediction
- Trained ML model integration

## 🛠 Tech Stack
- Python
- Streamlit
- Pandas
- Scikit-learn

## 📁 Project Files
- app.py – Streamlit application
- model.pkl – trained ML model
- requirements.txt – dependencies




