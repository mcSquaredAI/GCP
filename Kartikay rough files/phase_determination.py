


# Importing the libraries 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

# Importing the dataset 
datas =  cntr_plc01
datas 


X = datas.index.values.reshape(-1, 1)
y = datas.iloc[:, 0].values 
y = datas.loc[:, ['Canada']].values

# Fitting Linear Regression to the dataset 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 

lin.fit(X, y) 



# Fitting Polynomial Regression to the dataset 
from sklearn.preprocessing import PolynomialFeatures 

poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 

# poly.get_feature_names("x0")


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
# lin.predict(110.0) 

# Predicting a new result with Polynomial Regression 
lin2.predict(poly.fit_transform([[110.0]])) 




coefs = lin2.coef_
lin2.intercept_

X




def slope(X, coefs ):
    # print(x)
    # print(coefs)
    rt = [0]*len(X)
    for inx0,x in enumerate(X):
        for inx, cf in enumerate(coefs):
            # print(inx)
            # if inx == 0:
            #     continue
            rt[inx0] += (inx)*cf*(x** np.max([0, (inx-1)]) )
    
    return rt

cntr_plc0 = {}

for cntry in countries:
    # cntry = countries[0]
    tmp_frame = copy.deepcopy( plc.loc[:, [(cntry)]] )
    cntr_plc0[cntry] = tmp_frame.sum( axis =1).values

cntr_plc01 = pd.DataFrame.from_dict( cntr_plc0 )

datas =  copy.copy( cntr_plc01 )
X = datas.index.values.reshape(-1, 1)
# y = datas.iloc[:, 0].values 
y = datas.loc[:, ['Canada']].values


poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 

# poly.get_feature_names("x0")


poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 





# Visualising the Polynomial Regression results 
plt.scatter(X, y/90, color = 'blue') 
plt.plot(X, lin2.predict(poly.fit_transform(X))/np.array([[90]]), color = 'red') 
plt.plot(X, slope(X, lin2.coef_[0]) , color = 'green') 
plt.plot(X, [0]*len(y) , color = 'black') 
plt.title('Polynomial Regression') 
plt.xlabel('Temperature') 
plt.ylabel('Pressure') 

plt.show() 

plt.close('all')
# def slope(X, coefs ):
# 
#     return [coefs[1] + 2*coefs[2]*(x) + 3*coefs[3]*(x**2) + 4*coefs[4]*(x**3) for x in X]
#      

# 
# slope( 150, lin2.coef_[0] )
# 
# 
# 
# # Visualising the Polynomial Regression results 
# plt.scatter(X, y, color = 'blue') 
# 
# plt.plot(X, lin2.predict(poly.fit_transform(X)), color = 'red') 
# plt.title('Polynomial Regression') 
# plt.xlabel('Temperature') 
# plt.ylabel('Pressure') 
# 
# plt.show() 
# 

cntr_plc0 = {}

for cntry in countries:
    # cntry = countries[0]
    tmp_frame = copy.deepcopy( plc.loc[:, [(cntry)]] )
    cntr_plc0[cntry] = tmp_frame.sum( axis =1).values

cntr_plc01 = pd.DataFrame.from_dict( cntr_plc0 )

tmp_frame = copy.deepcopy( plc.loc[:, [('Canada')]] )
cntr_plc01 = tmp_frame

# Importing the dataset 
datas =  cntr_plc01
datas 


X = datas.index.values.reshape(-1, 1)
y = datas.iloc[:, 0].values 
# y = datas.loc[:, ['Canada']].values

for iy in range(6):
    # iy = 1
    y = datas.iloc[:, iy].values
    
    poly = PolynomialFeatures(degree = 4) 
    X_poly = poly.fit_transform(X) 
    
    # poly.get_feature_names("x0")
    
    
    poly.fit(X_poly, y) 
    lin2 = LinearRegression() 
    lin2.fit(X_poly, y) 
    
    
    
    
    
    # Visualising the Polynomial Regression results 
    plt.scatter(X, y/90, color = 'blue') 
    plt.plot(X, lin2.predict(poly.fit_transform(X))/np.array([90]), color = 'red') 
    plt.plot(X, slope(X.reshape(1,-1)[0], lin2.coef_) , color = 'green') 
    plt.plot(X, [0]*len(y) , color = 'black') 
    plt.title('Polynomial Regression') 
    plt.xlabel('Temperature') 
    plt.ylabel('Pressure') 
    
    plt.show() 
    
    
