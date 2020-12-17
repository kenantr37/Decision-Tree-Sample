# -*- coding: utf-8 -*-
"""
Decision Tree Sample 

@author: Zeno
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
#let's make an example of prediction of the car price with x1= year
x = np.array([2,3,4,5,6,7]).reshape(-1,1) # year of the car
y = np.array([30,27,23,20,19,17]) #price of the car
#we need to fit our data into our tree model
tree_model = DecisionTreeRegressor().fit(x,y)
#we will compare predictions between Decisin Tree and others 
linear_reg_model = LinearRegression().fit(x,y) 
polynomial_reg_model = PolynomialFeatures(degree = 5).fit_transform(x, y)
#ı need to create linear model for using polynomial model
linearmodel_forPolynomial = LinearRegression().fit(polynomial_reg_model, y)
#but for decision tree , we want to see gradual graph
#in here , we used numpy arange method and min , max method to make graph gradualy
x_gradual = np.arange(min(x),max(x),0.01).reshape(-1,1)
#we need to predict our model now
y_head = tree_model.predict(x_gradual)
y_head_linear = linear_reg_model.predict(x)
y_head_polynomial = linearmodel_forPolynomial.predict(polynomial_reg_model) 
#we visualize our model
#consuquently ı made 3 models and Let's compare
plt.scatter(x,y,color = "green")
plt.plot(x_gradual,y_head,color="purple",label = "Decision Tree model")
plt.plot(x,y_head_linear,color = "orange",label ="linear regression model")
plt.plot(x,y_head_polynomial,color = "blue",label = "polynomial regression model")
plt.legend()
plt.xlabel("car's year")
plt.ylabel("car's price")
plt.grid(True)
plt.show()
