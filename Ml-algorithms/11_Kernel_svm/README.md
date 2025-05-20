## 📘 What is Kernel SVM with RBF?

**Kernel Support Vector Machines (SVM)** extend the basic linear SVM by enabling it to solve **non-linear classification problems** using a mathematical trick called the **kernel trick**.

In this project, we use the **Radial Basis Function (RBF)** kernel — the most common kernel for capturing **non-linear relationships**.

---

### 🧠 Key Concepts

#### 🔹 1. **Linear SVM Limitation**
- A **linear kernel** can only draw **straight-line (or hyperplane)** decision boundaries.
- For complex patterns like concentric circles or curved boundaries, a linear classifier fails.

#### 🔹 2. **Kernel Trick**
- The **kernel trick** allows SVM to work in **higher-dimensional spaces** without explicitly computing the coordinates.
- The RBF kernel computes the **similarity between data points** in this new space.

#### 🔹 3. **RBF Kernel Formula**

\[
K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right)
\]

- \( \gamma \): Controls the influence of individual training samples.
- Small \( \gamma \) → smooth, general boundary.  
- Large \( \gamma \) → tight, specific boundary (risk of overfitting).

---

### ⚙️ When to Use RBF Kernel

| Situation                     | RBF Kernel Usefulness     |
|------------------------------|---------------------------|
| Data is not linearly separable | ✅ Excellent               |
| Decision boundary is curved  | ✅ Captures complex patterns |
| Small to medium-sized dataset | ✅ Efficient               |

---

### ✅ Characteristics

| Property          | Value                             |
|------------------|-----------------------------------|
| Model Type        | Non-linear binary classifier      |
| Kernel            | RBF (Radial Basis Function)       |
| Handles            | Non-linear patterns               |
| Requires Scaling | ✅ Yes, very important             |
| Sensitivity       | Highly sensitive to `γ` and `C`   |

---

## 📂 Dataset

- File: `iphone_purchase_records.csv`
- Columns:
  - `Gender` (Male/Female)
  - `Age`
  - `EstimatedSalary`
  - `Purchased` (0 or 1)

## 🧠 Workflow

1. **Data Preprocessing**
   - Encode `Gender` using `LabelEncoder`
   - Optionally convert all features to float
   - Split dataset into training and test sets
   - Apply `StandardScaler` for feature scaling

2. **Model Training**
   - Train an `SVC` model from `sklearn.svm` with:
     - `kernel="rbf"`
     - `random_state=0`

3. **Prediction**
   - Predict outcomes for test data

4. **Model Evaluation**
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

- `iphone_kernel_svm.py`: Source code with preprocessing, model training, evaluation
- `iphone_purchase_records.csv`: Dataset (not included, add manually)
- `README.md`: Project overview
