
## 📘 What is K-Nearest Neighbors (KNN)?

**K-Nearest Neighbors (KNN)** is a **non-parametric**, **instance-based** machine learning algorithm used for **classification** (and also regression). It classifies a new data point based on the **majority class of its nearest neighbors** in the training set.

The KNN algorithm is simple but powerful — it makes no assumptions about the underlying data distribution.

### 🧠 How It Works

1. Choose the number of neighbors: `k` (e.g., 5).
2. Calculate the **distance** from the new point to all training points.
3. Identify the `k` nearest training points.
4. Assign the **most common class** among the neighbors to the new point.

### 📏 Distance Metrics

- **Euclidean Distance** (default):  
\[
\text{Distance} = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2 + \dots}
\]
- Controlled using:
  - `metric="minkowski"`
  - `p=2` for Euclidean, `p=1` for Manhattan distance

## 📂 Dataset

- File: `iphone_purchase_records.csv`
- Columns:
  - `Gender` (Male/Female)
  - `Age`
  - `EstimatedSalary`
  - `Purchased` (0 or 1)

## 🧠 Workflow

1. **Data Preprocessing**
   - Encode categorical variable `Gender` using `LabelEncoder`
   - Optionally convert inputs to float
   - Split data into training and test sets
   - Apply standardization using `StandardScaler`

2. **Model Training**
   - Fit a `KNeighborsClassifier` using:
     - `n_neighbors = 5`
     - `metric = minkowski`, `p = 2` (Euclidean Distance)

3. **Model Evaluation**
   - Predict test set outcomes
   - Evaluate with:
     - Confusion Matrix
     - Accuracy
     - Precision
     - Recall


## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies using:
```bash
pip install pandas numpy scikit-learn
```

## 📁 Files

- `iphone_knn_classifier.py`: Full implementation of preprocessing, training, and evaluation
- `iphone_purchase_records.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
