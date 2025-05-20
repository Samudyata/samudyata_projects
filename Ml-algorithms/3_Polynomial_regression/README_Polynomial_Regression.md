
## 📘 What is Polynomial Regression?

**Polynomial Regression** is a type of regression analysis where the relationship between the independent variable (`X`) and the dependent variable (`y`) is modeled as an **nth-degree polynomial**.

While Linear Regression fits a straight line:
\[
y = b_0 + b_1x
\]

Polynomial Regression fits a curve:
\[
y = b_0 + b_1x + b_2x^2 + b_3x^3 + \dots + b_nx^n
\]

This allows the model to capture complex, non-linear patterns in the data.

### 🧠 Why Use Polynomial Regression?

- When data shows **non-linear trends** that a straight line can't capture.
- Adds polynomial terms (like \(x^2\), \(x^3\), etc.) to allow curved predictions.

### 🔧 How It Works in Code

1. **Linear Regression:**
   - Fits a straight line to the data.
   - Often underfits if the relationship is non-linear.

2. **Polynomial Regression:**
   - Uses `PolynomialFeatures` to add powers of `X` (e.g., \(x^2\), \(x^3\), \(x^4\)).
   - Fits a linear regression model on the **transformed input** to learn curves.

### 🧪 Example:

For `X = [[6.5]]`, and `degree = 4`, the transformation becomes:

\[
X_{\text{poly}} = [1,\ 6.5,\ 6.5^2,\ 6.5^3,\ 6.5^4]
\]

This allows the model to fit a smooth curve through the dataset.

### ✅ When to Use

Use Polynomial Regression when:
- Linear regression is too simple.
- Your data has curvature or complex trends.
- You want a flexible model without switching to non-linear algorithms.



## 📂 Dataset

- File: `Position_Salaries.csv`
- Columns:
  - `Position` (ignored for modeling)
  - `Level` (used as input feature `X`)
  - `Salary` (target `y`)

## 🧠 Workflow

1. **Data Preprocessing**
   - Load data, isolate `Level` as independent variable and `Salary` as dependent variable

2. **Linear Regression**
   - Fit a simple linear model and predict salary
   - Visualize predictions vs actual data

3. **Polynomial Regression**
   - Transform input `X` using `PolynomialFeatures`
   - Fit and visualize Polynomial models for degree 2, 3, and 4
   - Predict salary for Level 6.5 under each polynomial degree

## ✅ Results

| Model Type          | Predicted Salary (Level 6.5) |
|---------------------|------------------------------|
| Linear Regression   | ~$330,000                    |
| Polynomial Degree 2 | ~$189,000                    |
| Polynomial Degree 3 | ~$133,000                    |
| Polynomial Degree 4 | ~$158,000 (most reasonable)  |

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

- `polynomial_regression.py`: Full source code for Linear and Polynomial Regression
- `Position_Salaries.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
