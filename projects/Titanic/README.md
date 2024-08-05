# Titanic

The good ol classic Titanic Dataset.

Split dataset into 80-20 train-test split, and performed standard Feature Engineering (Imputing of NaN values, One-Hot Encoding of Categorical Variables) using `sklearn` package in Python.

The models used in the dataset were:
1. Naive Bayes
2. Logistic Regression
3. Decision Tree
4. K Nearest Neighbors
5. AdaBoosting (Ensemble)
6. Random Forest (Ensemble)
7. Support Vector Machine
8. PCA & Logistic Regression
9. Majority Voting (Ensemble)
10. Stacking (Ensemble)

The Majority Voting Classifier (consisting of `SVM`, `Random Forest Classifier`, `K Nearest Neighbors`, `Logistic Regression` with `PCA` features, `Decision Tree`) was the most accurate model with an overall accuracy of 84.36%.
