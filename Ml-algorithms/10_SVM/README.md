## 📘 What is Support Vector Classification (SVC)?

**Support Vector Machines (SVM)** are powerful supervised learning models used for **binary classification**. The key idea is to find the **optimal decision boundary (hyperplane)** that best separates the two classes with the **maximum margin** — the greatest possible distance from the nearest data points on each side (called **support vectors**).

### 🧠 Key Concepts

- **Hyperplane**: A boundary that separates two classes
- **Support Vectors**: Critical data points closest to the hyperplane that define its position
- **Margin**: The distance between the hyperplane and the support vectors; SVM tries to **maximize** this margin
- **Kernel**: A function that transforms data into higher dimensions when classes aren’t linearly separable

> In this project, we use a **linear kernel**, meaning the model looks for a straight-line separator in feature space.

### ⚙️ How the SVC Classifier Works

1. **Feature Scaling** is essential since SVM is distance-based.
2. The SVM classifier finds the **maximum-margin hyperplane** separating class `0` and class `1`.
3. Predictions are made by checking which side of the hyperplane a new point lies on.

### 📏 Why Use `kernel="linear"`?

- Use a **linear kernel** when the classes appear to be **linearly separable** (i.e., you can draw a straight line to separate them).
- Other options include:
  - `rbf` for curved boundaries
  - `poly` for polynomial separation

### 📊 Evaluation Metrics

| Metric     | Description                                  |
|------------|----------------------------------------------|
| Accuracy   | Overall correct predictions                  |
| Precision  | Correct positives out of predicted positives |
| Recall     | Correct positives out of actual positives    |
| Confusion Matrix | True/false positives and negatives     |

### ✅ When to Use SVM

- Binary classification with **well-separated classes**
- You want a **robust model** that works well even in high-dimensional spaces
- Your dataset is **not too large** (SVMs can be slower on huge datasets)

## 📂 Dataset

- File: `iphone_purchase_records.csv`
- Columns:
  - `Gender` (Male/Female)
  - `Age`
  - `EstimatedSalary`
  - `Purchased` (0 or 1)

## 🧠 Workflow

1. **Data Preprocessing**
   - Encode the `Gender` column
   - Optionally convert features to float
   - Split the data into training and test sets
   - Standardize features using `StandardScaler`

2. **Model Training**
   - Use `SVC` from `sklearn.svm` with `kernel="linear"`

3. **Model Prediction**
   - Predict user purchase behavior on test data

4. **Model Evaluation**
   - Confusion Matrix
   - Accuracy Score
   - Precision Score
   - Recall Score

## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies using:
```bash
pip install pandas numpy scikit-learn
```

## 📁 Files

- `iphone_svm_classifier.py`: Source code for preprocessing, training, and evaluation using SVM
- `iphone_purchase_records.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
