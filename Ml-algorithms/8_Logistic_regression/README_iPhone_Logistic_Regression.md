## 📘 What is Logistic Regression?

**Logistic Regression** is a supervised machine learning algorithm used for **binary classification**. It predicts the probability that a given input belongs to one of two classes — in this case, whether a person will **purchase an iPhone** (`1`) or not (`0`).

Unlike Linear Regression which outputs continuous values, Logistic Regression uses a **sigmoid (S-shaped) function** to output probabilities between 0 and 1.

### 🧠 Sigmoid Function

\[
P(y = 1 \mid X) = \frac{1}{1 + e^{-(b_0 + b_1x_1 + b_2x_2 + \dots + b_nx_n)}}
\]

- \( P \) is the probability that the output is 1 (iPhone purchased)
- \( b_0, b_1, \dots \) are the model's coefficients
- If \( P > 0.5 \): predict class `1`  
- Else: predict class `0`

### ⚙️ Features Used

- `Gender` (encoded numerically)
- `Age`
- `EstimatedSalary`

These inputs are **scaled** before being passed to the model to improve performance.
## 📂 Dataset

- File: `iphone_purchase_records.csv`
- Columns:
  - `Gender` (Male/Female)
  - `Age`
  - `EstimatedSalary`
  - `Purchased` (0 or 1)

## 🧠 Workflow

1. **Data Preprocessing**
   - Convert gender to numerical format using `LabelEncoder`
   - Optionally convert input data to `float`
   - Split data into training and testing sets
   - Apply feature scaling with `StandardScaler`

2. **Model Training**
   - Fit `LogisticRegression` model using `liblinear` solver

3. **Model Evaluation**
   - Predict test set outcomes
   - Generate a confusion matrix
   - Evaluate accuracy, precision, and recall

4. **Prediction Scenarios**
   - Test model with various hypothetical cases:
     - Male/Female aged 21 or 41 with $40k or $80k salary

## ✅ Results

- Classification metrics such as **accuracy**, **precision**, and **recall** were printed to evaluate model performance.
- Example prediction:
> Male aged 41 making $80k → **Likely to purchase iPhone**

## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies using:
```bash
pip install pandas numpy scikit-learn
```

## 📁 Files

- `iphone_logistic_regression.py`: Source code for preprocessing, training, evaluation, and predictions
- `iphone_purchase_records.csv`: Dataset (not included, add manually)
- `README.md`: Project summary
