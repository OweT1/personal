# Stock Market Analysis

Stumbled upon this whilst looking through the IT1244 Projects. Decided to do this as it was labelled as 'Challenging'.
This is a Regression problem, where we are looking at predicting the 4 features, `Open`, `Close`, `High` and `Low` using a regression model.

I decided to go with a simple `LinearRegression()` model, since there were `Rating` features that would be useful in a Machine Learning model. Hence, the ARIMA model was not used.
Instead, a 5 day Simple Moving Average and 2 days worth of previous data for the 4 features were included as features for training the model, and thereafter predicting future values.
The data was partitioned by the respective stock symbol, before they were split into the training and testing data with a 80% train split, where the training data would be the first 80% of the respective datasets partitioned by the stock symbol.

## Files
- data.parquet/data.csv (.parquet format is good for efficient loading and saving)
    - The main dataset containing stock price, trade volume, news events and news sentiment for S&P 500 companies during the period Oct 2020-Jul 2022
    - 217811 samples in total
    - Total 26 features per sample
    - Prediction Task:
        - Focus on prediction the following 4 features: "Open" (opening price on the day), "Close" (closing price of the day), "High" (highest price on that day), "Low" (lowest price on that day). If you predicting for day X, then you cannot use any of these 4 feature values of day X as model input
        - If you are predicting for day X, then you should also use some attributes of previous N (experiment with different values of N) days (remember that this is a time series task -> prediction of today is also affected by the occurrences of the near past). These attributes may include "Open", "Close", "High" and "Low" features as well.
- sp500wiki.parquet/sp500wiki.csv
    - List of S&P 500 companies as of July 2022 and various metadata in tabular format
    - Contains information for over 500 companies (524 rows in total)
    - 10 attributes per company (10 columns)
    - You can use these attributes as assisting feature when performing prediction task on a particular day for a particular company
 ## Additional Information:
- "Symbol" which denotes the company brand, is the common feature between the two datasets
- Missing values are there in the dataset
- Categorical, discrete and continuous attributes exist in the dataset
- The prediction tasks are regression tasks
- Make sure that train and test data have minimum information leakage (this is something you need to think about deeply)  
