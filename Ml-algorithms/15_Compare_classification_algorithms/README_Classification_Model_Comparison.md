
# 📊 iPhone Purchase – Classification Model Comparison

This project compares the performance of six popular classification algorithms using **10-fold cross-validation** to predict whether a user is likely to purchase an iPhone.

## 📌 Objective

To determine the most effective classification algorithm based on accuracy and standard deviation of performance metrics.

## 📂 Dataset

- File: `iphone_purchase_records.csv`
- Features:
  - `Gender` (categorical, encoded)
  - `Age`
  - `EstimatedSalary`
  - `Purchased` (target: 0 or 1)

## 🧠 Workflow

1. **Data Preprocessing**
   - Encode `Gender` using `LabelEncoder`
   - Standardize all features using `StandardScaler`

2. **Model Comparison**
   - Use `KFold` with 10 splits
   - Evaluate each model with `cross_val_score`
   - Metrics reported:
     - Mean Accuracy
     - Standard Deviation (SD) of Accuracy

3. **Models Evaluated**
   - Logistic Regression
   - K-Nearest Neighbors
   - Kernel SVM (RBF Kernel)
   - Naive Bayes
   - Decision Tree (Entropy)
   - Random Forest (Entropy, 100 Trees)

## ✅ Results

| Model               | Mean Accuracy | Std Dev |
|---------------------|---------------|---------|
| Logistic Regression | 84.00%        | 6.24%   |
| K Nearest Neighbor  | 91.25%        | 5.15%   |
| Kernel SVM          | 90.75%        | 4.88%   |
| Naive Bayes         | 88.75%        | 5.15%   |
| Decision Tree       | 85.00%        | 7.07%   |
| Random Forest       | 88.75%        | 4.51%   |

✅ **K-Nearest Neighbors** achieved the highest accuracy with good consistency, closely followed by **Kernel SVM** and **Random Forest**.

## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies using:
```bash
pip install pandas numpy scikit-learn
```

## 📁 Files

- `classification_model_comparison.py`: Full code for preprocessing, model comparison, and evaluation
- `iphone_purchase_records.csv`: Dataset (not included, add manually)
- `README.md`: Project summary
