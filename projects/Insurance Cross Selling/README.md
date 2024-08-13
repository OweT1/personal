# Insurance Cross Selling

More details about the Kaggle Competition can be found here: [https://www.kaggle.com/competitions/playground-series-s4e7](https://www.kaggle.com/competitions/playground-series-s4e7)

The dataset can be found in the Kaggle Competition. It is not uploaded in the repo due to its large size.

Was inspired to do a Kaggle competition for fun and thought this would be cool.

Key Performance Metric: Area Under ROC Curve

Models Used:
| No. | Model               | Area Under ROC Curve |
|:---:|---------------------|:--------------------:|
| 1   | Logistic Regression | 0.500054             |
| 2   | Ridge Classifier    | 0.500016             |
| 3   | SGD Classifier      | 0.500000             |
| 4   | K Nearest Neighbors | 0.578301             |
| 5   | Naive Bayes         | 0.792266             |
| 6   | Decision Tree       | 0.622570             |
| 7   | Neural Network      | 0.657627             |

The Area Under ROC Curve above was derived using 20% of the overall training dataset, aka the validation dataset.

The Python Packages/Libraries Used: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `sklearn`, `PyTorch`
