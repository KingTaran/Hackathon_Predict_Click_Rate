# Hackathon_Predict_Click_Rate
To predict click target by taking input of 20+ features.


Steps followed while building the Model
1)Understanding Business Objective 
2)Understanding Data
3)Data Preprocessing/Exploratory Data Analysis/Feature Engineering
  1)Checking for Missing Values if Any.
  2)Checking variance of each column so as to finalize columns for pre-processing
  3)Now we find some categorical columns which are day_of_week,times_of_day,category,
    target_audience,product,sender and apply dummy variables on them.
  4)Then we moved on to doing dummy variables then removing original column and join the 
    dummy columns.
  5)Then we made numerous plots and saw various charts which gave us a idea on their distribution. 
  6)We came to know our target variable is skewed so we performed transformation on target variable. 
  7)Then we did Feature Engineering where we removed some columns with very less data and converted
    them into binary columns or boolean flags.
  8)Then at last we scaled our numerical columns.  
  9)Now our data set is ready to perform modelling.

4)Model Development
  1)Tried various models such as Linear Regression, DecisionTree Regression, RandomForest Regressor,
    Gradient Boosting Regressor, LGBMRegressor, XGBRFRegressor.
  2)RandomForest Regressor with QuantileTransformer on Target Variable gave best r2 value of 0.55.
  3)Tried gridsearchCV and randomsearch for hyperparameter tuning but r2 value didn't increase.
  4)Predicted Test data on the model.


 
