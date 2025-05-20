## 📘 What is Multiple Linear Regression?

**Multiple Linear Regression** is a supervised learning algorithm used to predict a continuous dependent variable based on **two or more** independent variables. It extends simple linear regression by considering the combined effect of multiple features.

## 🧠 Theory Summary

1. **Objective**
   - Model the relationship between one target variable (`Profit`) and several predictors (e.g., `R&D Spend`, `Marketing Spend`, `State`, etc.).

2. **Regression Equation**
   \[
   y = b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n + \varepsilon
   \]
   - \( y \): Predicted Profit  
   - \( x_1, x_2, ..., x_n \): Features (spending and state)  
   - \( b_0 \): Intercept (base profit)  
   - \( b_1, b_2, ..., b_n \): Coefficients (impact of each variable)  
   - \( \varepsilon \): Error term

3. **One-Hot Encoding**
   - Converts categorical variables (like `State`) into numeric dummy variables so that the model can process them.

4. **Dummy Variable Trap**
   - Occurs when one dummy variable is perfectly correlated with others.
   - Avoided by dropping one dummy column to prevent multicollinearity.

5. **Training the Model**
   - The model learns the optimal values of \( b_0, b_1, ..., b_n \) using **Ordinary Least Squares (OLS)** to minimize prediction error.

6. **Prediction**
   - Once trained, the model can predict `Profit` for new input data using the learned equation.

7. **Assumptions**
   - Linear relationship between inputs and output
   - No multicollinearity among predictors
   - Homoscedasticity (equal variance of errors)
   - Errors are normally distributed
   - Observations are independent

## 📈 Evaluation

- Model performance was evaluated using visual comparison between actual and predicted values.
- Backward Elimination was applied using p-values from OLS regression to identify significant features.


## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`
- `statsmodels`

Install dependencies using:
```bash
pip install pandas numpy scikit-learn statsmodels
```

## 📁 Files

- `multiple_linear_regression.py`: All steps from loading data to backward elimination
- `50_Startups.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
