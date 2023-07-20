import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
url = 'https://raw.githubusercontent.com/Rafig77/data/main/kc_house_data.csv'
df=pd.read_csv(url)
import seaborn as sns
from sklearn.metrics import r2_score
 
import matplotlib.pyplot as plt

#NOTE SELECT  rows you want to exucute accoring to the task and uncommet them via Ctrl + /
#Task1
# print(df.dtypes)

#Task2

# print(df.columns)
# df.drop('id', axis=1,inplace=True)
# print(df.describe())

# #Task3
# #We are getting unique values for floors and converting frame by to_frame
# df2=df['floors'].value_counts()
# df2.to_frame()
# print(df2)

# #Task4
# Use the function boxplot in the seaborn library to determine whether 
# houses with a waterfront view or without a waterfront view have more price outliers.  
# Price houese with water and without water
# sns.boxplot(x='waterfront',y='price',data=df)
# plt.show()
#Task5
# sns.regplot(x='sqft_above',y='price',data=df)
# plt.show()
#Task6
# Separate the feature and target variable
# X = df[['sqft_living']]
# y = df['price']
# # Create and fit the linear regression model
# model = LinearRegression()
# model.fit(X, y)
# # Predict the target variable
# y_pred = model.predict(X)
# # Calculate the R-squared score
# r2 = r2_score(y, y_pred)
# # Print the R-squared score
# print("R-squared score:", r2)


#Task7
# features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] ]    
# y = df['price']
# model=LinearRegression()
# model.fit(features,y)
# y_predict=model.predict(features)
# r2=r2_score(y,y_predict)
# print("R-squared score:", r2)

#Task8
# from sklearn.pipeline import Pipeline

# # Define the list of features
# features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] ]    
# y = df['price']

# # Create a pipeline object
# pipeline = Pipeline([
#     ('regressor', LinearRegression())
# ])

# # Fit the pipeline
# pipeline.fit(features, y)

# # Predict the target variable
# y_pred = pipeline.predict(features)

# # Calculate the R-squared score
# r2 = r2_score(y, y_pred)

# # Print the R-squared score
# print("R-squared score:", r2)

#Task 9 
from sklearn.linear_model import Ridge
# model=Ridge(alpha=0.1)
# features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] ]    
# y = df['price']
# model.fit(features,y)
# y_pred=model.predict(features)
# r2 = r2_score(y, y_pred)
# print("R-squared score:", r2)

#Task 10
# features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"] ]    
# y = df['price']
# pol_feature=PolynomialFeatures(degree=2)
# x_pol=pol_feature.fit_transform(features)
# model=Ridge(alpha=0.1)
# model.fit(x_pol,y)
# y_pred=model.predict(x_pol)
# r2 = r2_score(y, y_pred)
# print("R-squared score:", r2)