# Task-5-Heart-Disease-Classification-with-Decision-Tree-Random-Forest


This project compares two machine learning algorithms—Decision Tree and Random Forest—for predicting heart disease based on patient data.

Tasks Covered

1. Load and preprocess the dataset (`heart.csv`)
2. Train a Decision Tree Classifier and visualize it
3. Analyze overfitting by plotting accuracy vs tree depth
4. Train a Random Forest Classifier
5. Evaluate both models using:
   - Accuracy
   - Confusion Matrix
   - Precision, Recall, F1-Score
   - ROC-AUC
6. Analyze feature importance for Random Forest
7. Perform cross-validation for robust evaluation

Key Findings

- Random Forest outperformed Decision Tree in both accuracy and ROC-AUC.
- Overfitting in Decision Trees can be controlled using `max_depth`.
- Key features include `cp`, `thalach`, and `oldpeak`.

Files

- `heart.csv`: Dataset file (from [Kaggle](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset))
