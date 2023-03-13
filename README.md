# datagenie_hackathon
Time series Classification

The goal is to create an efficient time series model selection  algorithm.

We need to create an efficient time series classification algorithm so that if we give a new time series data it needs to classify which model is better for the given time series model and we need to generate predictions with help of the best model.

I have used the sample csv files which are given in the problem statement. The sample data contains different datasets which are daily, weekly, hourly, monthly. By using all these datasets we have to build different time series models.

First in Preprocessing steps I have filled the null values using interpolation and also change the datatypes if needed and also setting the index.

FeatureEngineering : In this we need to create new features which are useful for classification model.

For time series models I have used ARIMA SARIMAX and XGBoost models for fitting time series data. In XGBoost as we dont have features so we need more features so I have created features for that using create_features() function. After this I have created seperate functions for each model so that it will be easy for future use. Hyperparametr tuning was done using grid search algorithm for ARIMA(p,d,q).

I have used mean absolute percentage error(MAPE) as an evaluation metric.

For time series classification we need to create a new dataset which contains features and output variables. To create features we need to call the FeatureEngineering function for every sample so that it will create new features. Now for the Output variable we need to find MAPE values of 3 models for all 36 datasets. 

Then we need to find the minimum MAPE value in each sample and append them in the list. Now with these features and output variables we need to create a dataset.

Now when a new sample is uploaded by user first basic preprocessing will be done and features will be extracted with help of FeatureEngineering then we need to predict the best model by using the classifier model and after getting the best model we need to model this sample with the best model and predict the future data.

After completing the model I started doing backend using FastAPI. It was hard at the beginning to build the backend but eventually by browsing I have created a simple backend which takes a csv file as input and stores it in a local drive. Then use that csv file for further processes. Then I have created a simple UI using HTML. It just takes csv file and other inputs and prints the results in second page.

![Screenshot (6)](https://user-images.githubusercontent.com/102681460/224605692-3f1fd530-945c-4239-bc5b-4f19becaeccb.png)
