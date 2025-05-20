## 📘 What is Decision Tree Regression?

**Decision Tree Regression** is a non-linear model that splits the dataset into smaller regions based on feature values. It then predicts the output for each region as the **mean of the target values** in that region.

It works like a series of "if-else" decisions:
- If the input falls within a specific range (region), predict the average value for that region.
- This makes it well-suited for problems where the target variable changes abruptly rather than smoothly.


### ⚙️ Visualization Insight

Unlike Linear or Polynomial Regression:
- Decision Trees create a **stair-step plot** rather than a smooth curve.
- This can model sudden jumps in the target value, but lacks smoothness.

### ✅ When to Use

Use Decision Tree Regression when:
- The dataset has **non-linear, non-continuous patterns**
- You want a model that is **easy to interpret**
- You’re dealing with **small to medium-sized datasets**
- The output changes in **discrete steps** (e.g., pricing models, classification-like boundaries)

### ⚠️ Limitations

- **Prone to overfitting** on small datasets unless you limit depth or prune the tree
- Not smooth — may not generalize well to unseen data without regularization


## 📂 Dataset

- File: `Position_Salaries.csv`
- Columns:
  - `Position` (ignored)
  - `Level` (input feature `X`)
  - `Salary` (target `y`)

## 🧠 Workflow

1. **Data Preprocessing**
   - Load data using `pandas`
   - Select only the `Level` and `Salary` columns for modeling

2. **Model Training**
   - Fit a `DecisionTreeRegressor` from `sklearn.tree`
   - Use default or custom settings (e.g., `random_state=0`)

3. **Prediction**
   - Predict the salary for **Level 6.5**
   - Result: **$150,000**

4. **Visualization**
   - Plot prediction steps across a continuous range to show step-wise function

## 📌 Key Insight

Decision Tree Regression is **non-linear** and **non-continuous**, making it suitable for problems with abrupt changes or discrete jumps in target value.

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

- `decision_tree_regression.py`: Full implementation of model training, prediction, and visualization
- `Position_Salaries.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
