
# 🤖 Comparative Evaluation of Popular Machine Learning Classification Algorithms

This project benchmarks six widely used machine learning classification models using cross-validation to assess their performance on the iPhone purchase prediction dataset.

## 📌 Objective

To compare the classification accuracy and robustness of different ML models in predicting user purchase behavior based on demographic features.

## 🧠 Algorithms Compared

- Logistic Regression
- K-Nearest Neighbors
- Kernel SVM (RBF)
- Naive Bayes
- Decision Tree
- Random Forest

## ✅ Key Metrics

Each model was evaluated using **10-fold cross-validation**, reporting:
- **Mean Accuracy**
- **Standard Deviation** of Accuracy

## 📦 Requirements

- `pandas`
- `numpy`
- `scikit-learn`

Install using:
```bash
pip install pandas numpy scikit-learn
```

## 📁 Files

- `ml_algorithm_comparison.py`: Source code comparing ML classifiers
- `iphone_purchase_records.csv`: Dataset used (not included)
- `README.md`: Overview of the evaluation process
