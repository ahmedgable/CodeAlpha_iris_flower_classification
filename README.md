# 🌸 Iris Flower Classification App

A machine learning web application built with **Streamlit** and **Scikit-Learn** that predicts the species of an Iris flower (*Setosa*, *Versicolor*, or *Virginica*) based on its sepal and petal measurements.

---

## 📌 Project Overview

This project implements an end-to-end Machine Learning pipeline:
1. **Data Preprocessing & EDA:** Feature scaling using `StandardScaler` and class label encoding with `LabelEncoder`.
2. **Model Training & Evaluation:** Trained and benchmarked multiple machine learning algorithms (Logistic Regression, Decision Trees, Random Forest, KNN, Naive Bayes, Gradient Boosting, SVM, etc.).
3. **Best Model Selection:** **Support Vector Machine (SVM)** achieved **100% accuracy** on the test dataset and was selected for production deployment.
4. **Interactive Web App:** Built a modern, user-friendly interface using **Streamlit** that displays species prediction along with confidence probabilities.

---

## 🚀 Interactive Web App Demo

The application allows users to input the following measurements:
- **Sepal Length (cm)**
- **Sepal Width (cm)**
- **Petal Length (cm)**
- **Petal Width (cm)**

It outputs:
- **Predicted Species:** (*Iris-setosa*, *Iris-versicolor*, or *Iris-virginica*)
- **Prediction Confidence:** Class probabilities with visual progress bars.

---

## 📊 Dataset Description

The famous **Iris Dataset** consists of 150 instances with 4 numeric features and 3 target classes:
- `Iris-setosa`
- `Iris-versicolor`
- `Iris-virginica`

---

## 🛠️ Project Structure

```text
├── app.py                      # Streamlit application script
├── iris_classification.ipynb   # Jupyter Notebook containing Data Analysis, Model Comparisons & Benchmarking
├── scaler.pkl                  # Fitted StandardScaler object
├── svm_model.pkl               # Trained SVM Classifier model
├── species_encoder.pkl         # Fitted LabelEncoder for target classes
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
