# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:31:24 2020


"""

#salary prediction using linear regression algorithm


import numpy as np
import pandas as pd
import matplotlib.pyplot as mpt


data_set= pd.read_csv('Salary_Data.csv')  
x=data_set.iloc[:,:-1] # independent variable
y=data_set.iloc[:,1]
# Splitting the dataset into training and test set.  
from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 1/3, random_state=0)  
#Fitting the Simple Linear Regression model to the training dataset  
from sklearn.linear_model import LinearRegression  
regression= LinearRegression()  
regression.fit(x_train, y_train)  
#Prediction of Test and Training set result  
y_pred= regression.predict(x_test)  
x_pred= regression.predict(x_train) 
mpt.scatter(x_train, y_train, color="green")   
mpt.plot(x_train, x_pred, color="red")    
mpt.title("Salary vs Experience (Training Dataset)")  
mpt.xlabel("Years of Experience")  
mpt.ylabel("Salary(In Rupees)")  
mpt.show()   
#visualizing the Test set results  
mpt.scatter(x_test, y_test, color="blue")   
mpt.plot(x_train, x_pred, color="red")    
mpt.title("Salary vs Experience (Test Dataset)")  
mpt.xlabel("Years of Experience")  
mpt.ylabel("Salary(In Rupees)")  
mpt.show()  
#combined rmse value
rss=((y_test-y_pred)**2).sum()
mse=np.mean((y_test-y_pred)**2)
print("Final rmse value is =",np.sqrt(np.mean((y_test-y_pred)**2)))
