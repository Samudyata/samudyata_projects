
## 🌲 What is Random Forest Regression?

**Random Forest Regression** is an **ensemble learning technique** that builds multiple decision trees and combines their outputs to improve prediction accuracy and reduce overfitting.

Unlike a single Decision Tree (which can be unstable or overly sensitive to the data), Random Forest:
- Builds **many trees** on different random subsets of the data
- Averages their predictions to produce a **more stable and accurate result**

### 🧠 How It Works

1. Random Forest creates **n estimators** (e.g., 100 trees), each trained on a random sample of the data (using bootstrapping).
2. At each node in each tree, it selects a **random subset of features** to find the best split — increasing diversity among trees.
3. For regression tasks, the final prediction is the **average of all tree predictions**.

### 🧪 Why Use Random Forest?

- Handles **non-linear and complex relationships**
- **Less prone to overfitting** than a single Decision Tree
- **Robust** to noise and outliers
- Works well even with small datasets

## 📂 Dataset

- File: `Position_Salaries.csv`
- Columns:
  - `Position` (ignored for modeling)
  - `Level` (used as input feature `X`)
  - `Salary` (target `y`)

## 🧠 Workflow

1. **Data Preprocessing**
   - Load data using `pandas`
   - Extract `Level` as `X` and `Salary` as `y`

2. **Model Training**
   - Fit `RandomForestRegressor` with:
     - 10 Trees
     - 100 Trees
     - 300 Trees
   - Visualize predictions using high-resolution grid

3. **Prediction**
   - Predict salary for `Level 6.5` using each model

## ✅ Results

| Model Type              | Predicted Salary (Level 6.5) |
|-------------------------|------------------------------|
| Random Forest (10 trees)| ~$167,000                    |
| Random Forest (100)     | ~$158,000                    |
| Random Forest (300)     | ~$160,000                    |
| Polynomial Regression   | ~$158,000                    |
| Support Vector Regression| ~$170,000                   |
| Decision Tree Regression| ~$150,000                    |

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

- `random_forest_regression.py`: Full implementation of model training, prediction, and visualization
- `Position_Salaries.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
