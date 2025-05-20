
# 🤖 Support Vector Regression – Predicting Salary by Position Level

This project implements **Support Vector Regression (SVR)** to predict the **Salary** of an employee based on their **Position Level**. We use the same dataset as in the Polynomial Regression project to compare results.

## 📌 Objective

To predict the salary for a candidate at **Level 6.5** using:
- Support Vector Regression (SVR)

## 📂 Dataset

- File: `Position_Salaries.csv`
- Columns:
  - `Position` (ignored)
  - `Level` (input feature `X`)
  - `Salary` (target `y`)

## 🧠 Workflow

1. **Data Preprocessing**
   - Load data, select `Level` and `Salary` columns
   - Apply **Feature Scaling** to both `X` and `y`

2. **Model Training**
   - Fit an SVR model using an **RBF kernel** from `sklearn.svm`

3. **Visualization**
   - Plot actual salary vs. predicted salary using scaled `X`

4. **Prediction**
   - Predict the salary for **Level 6.5**
   - Inverse transform the result to get actual dollar amount

## ✅ Results

The SVR model predicted the salary for Level 6.5 as **~$170,000**  
(Compared to ~$158,000 with Polynomial Regression)

## 📦 Requirements

- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install dependencies using:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## 📁 Files

- `svr_salary_prediction.py`: Full code for SVR model training, prediction, and visualization
- `Position_Salaries.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
