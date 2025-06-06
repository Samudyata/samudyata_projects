# Step 1 - Load Data
import pandas as pd
dataset = pd.read_csv("iphone_purchase_records.csv")
X = dataset.iloc[:, :-1].values  # All columns except last
y = dataset.iloc[:, 3].values    # Last column is the target

# Step 2 - Convert Gender to number
from sklearn.preprocessing import LabelEncoder
labelEncoder_gender = LabelEncoder()
X[:, 0] = labelEncoder_gender.fit_transform(X[:, 0]).astype(float)

# Step 2.5 - Ensure all data is float
import numpy as np
X = X.astype(float)

# Step 3 - Split Data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Step 4 - Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Step 5 - Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0, solver="liblinear")
classifier.fit(X_train, y_train)

# Step 6 - Predict
y_pred = classifier.predict(X_test)

# Step 7 - Confusion Matrix and Evaluation
from sklearn import metrics
cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy score:", accuracy)

precision = metrics.precision_score(y_test, y_pred)
print("Precision score:", precision)

recall = metrics.recall_score(y_test, y_pred)
print("Recall score:", recall)

# Step 8 - Make New Predictions
x1 = sc.transform([[1, 21, 40000]])
x2 = sc.transform([[1, 21, 80000]])
x3 = sc.transform([[0, 21, 40000]])
x4 = sc.transform([[0, 21, 80000]])
x5 = sc.transform([[1, 41, 40000]])
x6 = sc.transform([[1, 41, 80000]])
x7 = sc.transform([[0, 41, 40000]])
x8 = sc.transform([[0, 41, 80000]])

print("Male aged 21 making $40k will buy iPhone:", classifier.predict(x1))
print("Male aged 21 making $80k will buy iPhone:", classifier.predict(x2))
print("Female aged 21 making $40k will buy iPhone:", classifier.predict(x3))
print("Female aged 21 making $80k will buy iPhone:", classifier.predict(x4))
print("Male aged 41 making $40k will buy iPhone:", classifier.predict(x5))
print("Male aged 41 making $80k will buy iPhone:", classifier.predict(x6))
print("Female aged 41 making $40k will buy iPhone:", classifier.predict(x7))
print("Female aged 41 making $80k will buy iPhone:", classifier.predict(x8))

# Step 8 - Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Only use Age and Salary for 2D visualization
X_vis = X[:, 1:].astype(float)  # Age, Salary only
y_vis = y.astype(int)

# Train classifier again on Age and Salary
X_train_vis, X_test_vis, y_train_vis, y_test_vis = train_test_split(X_vis, y_vis, test_size=0.25, random_state=0)
sc_vis = StandardScaler()
X_train_vis = sc_vis.fit_transform(X_train_vis)
X_test_vis = sc_vis.transform(X_test_vis)

classifier_vis = LogisticRegression(random_state=0, solver="liblinear")
classifier_vis.fit(X_train_vis, y_train_vis)

# Plotting decision boundary
X_set, y_set = X_train_vis, y_train_vis
X1, X2 = np.meshgrid(
    np.arange(X_set[:, 0].min() - 1, X_set[:, 0].max() + 1, 0.01),
    np.arange(X_set[:, 1].min() - 1, X_set[:, 1].max() + 1, 0.01)
)

plt.figure(figsize=(10, 6))
plt.contourf(
    X1, X2,
    classifier_vis.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
    alpha=0.75,
    cmap=ListedColormap(('red', 'green'))
)
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# Plot training points
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(
        X_set[y_set == j, 0], X_set[y_set == j, 1],
        c=ListedColormap(('red', 'green'))(i), label=f"Purchased={j}"
    )

plt.title("Logistic Regression (Age vs Salary)")
plt.xlabel("Age (scaled)")
plt.ylabel("Salary (scaled)")
plt.legend()
plt.grid(True)
plt.show()

