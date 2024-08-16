# Titanic

The good ol classic Titanic Dataset.

Split dataset into 80-20 train-test split, and performed standard Feature Engineering (Imputing of NaN values, One-Hot Encoding of Categorical Variables) using `sklearn` package in Python.

Key Performance Metric: Accuracy

The models used in the dataset were:
| No. | Model                           | Accuracy (%) |
|:---:|---------------------------------|:------------:|
| 1   | Naive Bayes                     | 40.78        |
| 2   | Logistic Regression             | 78.77        |
| 3   | Decision Tree                   | 77.65        |
| 4   | K Nearest Neighbors             | 80.45        |
| 5   | AdaBoost                        | 78.21        |
| 6   | Random Forest                   | 82.12        |
| 7   | Support Vector Machine          | 81.56        |
| 8   | Logistic Regression (after PCA) | 79.33        |
| 9   | Majority Voting                 | 84.36        |
| 10  | Stacking                        | 81.56        |

The Majority Voting Classifier (consisting of `SVM`, `Random Forest Classifier`, `K Nearest Neighbors`, `Logistic Regression` with `PCA` features, `Decision Tree` models) was the most accurate model with an overall accuracy of 84.36%.
