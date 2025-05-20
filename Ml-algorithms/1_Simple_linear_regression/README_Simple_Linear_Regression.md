
# 📈 Simple Linear Regression – Salary Prediction

## 📘 What is Simple Linear Regression?

**Simple Linear Regression** models the relationship between two variables:

- **X (independent variable)** – for example, *Years of Experience*
- **y (dependent variable)** – for example, *Salary*

It fits a straight line (best-fit line) through the data using the equation:

\[
y = b_0 + b_1x
\]

Where:
- \( b_0 \) is the **intercept** (salary when experience is 0)
- \( b_1 \) is the **slope** (how much salary increases with each additional year of experience)

This helps us understand and predict how changes in experience affect salary.


This project implements a **Simple Linear Regression** model to predict the **Salary** of an employee based on their **Years of Experience**. The dataset is sourced from the "A-Z Machine Learning" course on Udemy and contains data for 30 employees in a company.

## 📌 Objective

To build and evaluate a simple linear regression model to find a correlation between:
- Independent variable: Years of Experience
- Dependent variable: Salary

## 📂 Dataset

- File: `Salary_Data.csv`
- Columns:
  - `Years of Experience`
  - `Salary` (target)

## 🧠 Workflow

1. **Data Preprocessing**
   - Load the dataset
   - Separate independent (`X`) and dependent (`y`) variables

2. **Model Building**
   - Split dataset into training and test sets (2:1 ratio)
   - Fit a linear regression model using `sklearn`

3. **Prediction**
   - Predict salary values on the test set
   - Compare with actual test results

4. **Visualization**
   - Plot regression line over both training and test data
   - Visualize the fit and spread of predictions

5. **New Prediction**
   - Predict salary for unseen experience values (e.g. 15 years)

## ✅ Outcome

The model shows a strong linear correlation between years of experience and salary. Visualization confirms a good fit for both training and testing sets.

## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

Install dependencies using:
```bash
pip install pandas numpy scikit-learn matplotlib
```

## 📁 Files

- `simple_linear_regression.py`: Full source code for loading, training, predicting, and visualizing
- `Salary_Data.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
