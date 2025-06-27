# 🧠 Probabilistic Topic Modeling & Clustering of E-Commerce Reviews

This project performs unsupervised Natural Language Processing (NLP) on a large dataset of women’s clothing reviews. It uses **Latent Dirichlet Allocation (LDA)** for topic modeling and **K-means** for customer segmentation, providing insights into customer concerns like *fit*, *style*, and *quality*.

> 📍 [Run the Code on Google Colab](https://colab.research.google.com/drive/1il_lAbPve3sGKe--FeyDdYnTcwZ7hMah?usp=sharing)

---

## 📁 FinalRequirements

Before running the Colab notebook, make sure you have a folder named `FinalRequirements` in your Google Drive that contains the pre-trained saved models.

---

## 🔍 Project Overview

- **Dataset**: [Kaggle - Women’s Clothing E-Commerce Reviews](https://www.kaggle.com/datasets/nicapotato/womens-ecommerce-clothing-reviews/data)
- **Techniques Used**:
  - Data cleaning and lemmatization with NLTK
  - Feature engineering (word counts, sentiment scores via VADER)
  - Topic modeling using LDA (via Gensim)
  - K-means clustering with PCA and t-SNE visualization
- **Libraries**: `nltk`, `gensim`, `sklearn`, `matplotlib`, `seaborn`, `pandas`, `vaderSentiment`

---

## 📊 Key Results

- **Clustering**:
  - K = 3 was optimal (silhouette score ≈ 0.53)
  - Cluster 0: Fit issues
  - Cluster 1: Style-positive reviews
  - Cluster 2: Quality concerns

- **Topics Identified**:
  - **Topic 0**: Dresses & Skirts
  - **Topic 1**: Jeans & Pants
  - **Topic 2**: Sweaters & Colors
  - **Topic 3**: Shirts & Sizing


---

## 🧪 How to Run

1. Open the [Colab Notebook](https://colab.research.google.com/drive/1il_lAbPve3sGKe--FeyDdYnTcwZ7hMah?usp=sharing)
2. Mount your Google Drive.
3. Ensure the `FinalRequirements` folder is accessible.
4. Run each cell sequentially.

---

## 📈 Visual Outputs

- PCA plots for clustering validation
- t-SNE visualizations of review segments
- PyLDAvis for interactive topic inspection

---

