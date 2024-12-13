# DATA Final Project - Team Loading...

## CART - Classification and Regression Tree for Classification

This project entails the implementation of the CART (Classification and Regression Tree) algorithm tailored for classification tasks. Our custom implementation successfully replicates the performance of Scikit-Learn's `DecisionTreeClassifier`, ensuring accuracy and reliability.

---

## Table of Contents

1. [Team Members](#team-members)
2. [GitHub Repository](#github-repository)
3. [Environment Setup](#environment-setup)
4. [Project Overview](#project-overview)
5. [Results](#results)
6. [References](#references)

---

## Team Members

- **Zhangchi Fan**  
  Email: [zhangchi_fan@brown.edu](mailto:zhangchi_fan@brown.edu)

- **Kexin Lin**  
  Email: [kexin_lin@brown.edu](mailto:kexin_lin@brown.edu)

- **Yixuan Wang**  
  Email: [yixuan_wang6@brown.edu](mailto:yixuan_wang6@brown.edu)

- **Zhecheng Zhang**  
  Email: [zhecheng_zhang@brown.edu](mailto:zhecheng_zhang@brown.edu)

---

## GitHub Repository

Access the project repository here:  
[https://github.com/Drowsywolf/data2060_final_project](https://github.com/Drowsywolf/data2060_final_project)

---

## Environment Setup

- **Python Version:** `3.12.5`
- **Dependencies:** Listed in `environment.yml`

To set up the environment, execute the following commands:

```bash
conda env create -f environment.yml
conda activate data2060
```

---

## Project Overview

### What is CART?

Classification and Regression Trees (CART) is a supervised machine learning algorithm used for classification and regression tasks. It builds a binary decision tree by recursively partitioning the feature space to maximize the purity of the child nodes using criteria like Gini impurity.

### How CART Works for Classification

1. **Tree Construction:**
   - **Splitting Criteria:** Utilizes Gini impurity to select the best feature and threshold for splitting.
   - **Recursive Partitioning:** Continuously splits the dataset until stopping criteria are met (e.g., maximum depth).

2. **Tree Pruning:**
   - Implements strategies like pre-pruning (limiting tree depth) to prevent overfitting.

3. **Prediction:**
   - Traverses the tree based on input features to assign class labels at leaf nodes.

### Advantages and Disadvantages

- **Advantages:**
  - **Interpretability:** Easy to visualize and understand decision paths.
  - **Handling Various Data Types:** Suitable for both numerical and categorical data.
  - **Feature Selection:** Automatically identifies the most informative features.

- **Disadvantages:**
  - **Prone to Overfitting:** Especially with deep trees.
  - **Instability:** Small data variations can lead to different tree structures.
  - **Bias Towards Features with More Levels:** May overlook equally important features with fewer categories.

---

## Results

### Reproducing Previous Work (`max_depth=3`)

Both our custom CART implementation and Scikit-Learn's `DecisionTreeClassifier` were evaluated using the same dataset and hyperparameters.

#### Classification Reports

| Class  | Precision | Recall | F1-Score | Support |
|--------|-----------|--------|----------|---------|
| acc    | 0.56      | 0.57   | 0.56     | 129     |
| good   | 0.00      | 0.00   | 0.00     | 20      |
| unacc  | 0.87      | 0.97   | 0.92     | 397     |
| vgood  | 0.00      | 0.00   | 0.00     | 25      |
| **Accuracy** |       |        | **0.80** | **571** |
| **Macro Avg** | 0.36 | 0.38 | 0.37 | 571 |
| **Weighted Avg** | 0.73 | 0.80 | 0.77 | 571 |

#### Accuracy Scores

| Metric       | Scikit-Learn `DecisionTreeClassifier` | Our CART Implementation |
|--------------|----------------------------------------|-------------------------|
| **Accuracy** | 80.21%                                 | 80.21%                  |

#### Confusion Matrices

Both models produced identical confusion matrices, demonstrating consistent classification performance.

|          | Predicted acc | Predicted good | Predicted unacc | Predicted vgood |
|----------|---------------|----------------|-----------------|-----------------|
| **acc**     | 73            | 0              | 56              | 0               |
| **good**    | 0             | 0              | 0               | 0               |
| **unacc**   | 12            | 0              | 385             | 0               |
| **vgood**   | 25            | 0              | 0               | 0               |

### Small Adjustment (`max_depth=1000`)

To explore deeper tree structures, we increased the `max_depth` to 1000 while keeping other hyperparameters constant.

- **Accuracy:** Improved to **81.44%**
- **Classification Metrics:** Enhanced precision and recall for minority classes.
- **Confusion Matrix:** Reflects better classification of previously underrepresented classes.

---

## References

- **Scikit-Learn:** [Decision Tree Classifier](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)  
  *Accessed on 8 November 2024*

- **Kaggle Tutorial:**  
  Banerjee, P. (2019). *Decision-Tree Classifier Tutorial*. Version 4.  
  [https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial](https://www.kaggle.com/code/prashant111/decision-tree-classifier-tutorial)  
  *Accessed on 11 December 2024*