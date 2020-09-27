to get country_product_lvl PLC just agregate from PLC and if need be normalise using population from features_agregate_10

First, Get country and ibndication clusters somehow:

be ready with 4-degree poly curves
find the growth of rate and determine phases
then align different curves in different phases and visualise them

# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
datas = pd.read_csv('data.csv') 
datas 

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 

lin.fit(X, y) 


# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 

poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 

poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 

# Visualising the Linear Regression results 
plt.scatter(X, y, color = 'blue') 

plt.plot(X, lin.predict(X), color = 'red') 
plt.title('Linear Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 

plt.show() 


# Visualising the Polynomial Regression results 
plt.scatter(X, y, color = 'blue') 

plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
plt.title('Polynomial Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 

plt.show() 


# Predicting a new result with Linear Regression 
lin.predict(110.0) 

# Predicting a new result with Polynomial Regression 
lin2.predict(poly.fit_transform(110.0)) 











