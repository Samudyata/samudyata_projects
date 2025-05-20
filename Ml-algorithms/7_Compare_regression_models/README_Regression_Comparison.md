
# 📚 Regression Model Comparison – Predicting Salary at Level 6.5

This project compares six fundamental regression models to predict an employee’s salary based on their position level. All models are trained using the same dataset to provide a direct performance comparison.

## 📌 Objective

A candidate for the **Regional Manager** role claims they are earning **$160,000**. We want to verify this claim using various regression models trained on the company’s internal dataset.

## 📂 Dataset

- File: `Position_Salaries.csv`
- Columns:
  - `Position` (ignored)
  - `Level` (input feature `X`)
  - `Salary` (target `y`)

## 🧠 Models Used

1. **Simple Linear Regression**
2. **Polynomial Regression** (Degree 4)
3. **Support Vector Regression (SVR)** with RBF kernel
4. **Decision Tree Regression**
5. **Random Forest Regression (300 Trees)**
6. **Comparison Summary + Visualization**

## ✅ Results

| Regression Model            | Predicted Salary at Level 6.5 |
|-----------------------------|-------------------------------|
| Linear Regression           | ~$330,000                     |
| Polynomial Regression (D=4) | ~$158,000                     |
| Support Vector Regression   | ~$170,000                     |
| Decision Tree Regression    | ~$150,000                     |
| Random Forest (300 Trees)   | ~$160,000                     |

✅ Based on the results, **Polynomial Regression** and **Random Forest Regression** provided the most reasonable predictions near $160K — suggesting the candidate might be truthful.

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

- `regression_comparison.py`: Full source code for data loading, model training, predictions, and visualization
- `Position_Salaries.csv`: Dataset (not included, add manually)
- `README.md`: Project summary and evaluation
