## 📘 What is Naive Bayes?

**Naive Bayes** is a **probabilistic classifier** based on **Bayes' Theorem**. It assumes that:
- All features are **independent** given the class label (this is the "naive" part).
- Features follow a **probability distribution**, such as Gaussian (Normal), Bernoulli, or Multinomial.

In this project, we use **Gaussian Naive Bayes**, which assumes that the **continuous features (like age and salary)** follow a **normal distribution**.

---

### 🧠 Bayes' Theorem

\[
P(y \mid X) = \frac{P(X \mid y) \cdot P(y)}{P(X)}
\]

Where:
- \( P(y \mid X) \): Posterior probability (e.g., will buy given age & salary)
- \( P(X \mid y) \): Likelihood of data given the class
- \( P(y) \): Prior probability of the class
- \( P(X) \): Probability of the data (can be ignored during prediction)

---

### 🔹 Gaussian Naive Bayes

For **continuous input features**, we assume each feature \( x \) follows a normal distribution:

\[
P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_y^2}} \exp\left( -\frac{(x_i - \mu_y)^2}{2\sigma_y^2} \right)
\]

Where:
- \( \mu_y \) and \( \sigma_y \) are the mean and standard deviation of feature \( x_i \) for class \( y \)

The model calculates this probability for each class and selects the class with the **highest posterior probability**.

---

## 📂 Dataset

- File: `iphone_purchase_records.csv`
- Columns:
  - `Gender` (categorical)
  - `Age`
  - `EstimatedSalary`
  - `Purchased` (target: 0 or 1)

## 🧠 Workflow

1. **Data Preprocessing**
   - Encode `Gender` using `LabelEncoder`
   - Convert data to float if necessary
   - Split data into training and test sets
   - Apply feature scaling using `StandardScaler`

2. **Model Training**
   - Train a `GaussianNB` classifier from `sklearn.naive_bayes`

3. **Model Prediction**
   - Predict outcomes on test set

4. **Model Evaluation**
   - Confusion Matrix
   - Accuracy Score
   - Precision Score
   - Recall Score



## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`

Install dependencies with:
```bash
pip install pandas numpy scikit-learn
```

## 📁 Files

- `iphone_naive_bayes.py`: Source code for preprocessing, training, and evaluation
- `iphone_purchase_records.csv`: Dataset (not included)
- `README.md`: Project summary and instructions
