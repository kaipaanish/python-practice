import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset =  pd.read_csv(r"C:\Users\anish\Downloads\emp_sal.csv")

x = dataset.iloc[:,1:2].values
y = dataset.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)


plt.scatter(x,y, color ="red")
plt.plot(x, lin_reg.predict(x), color ='blue')
plt.title("Linear regression model")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
lin_model_pred


from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures()
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

plt.scatter(x,y, color ="red")
plt.plot(x, lin_reg_2.predict(x_poly), color ='blue')
plt.title("poly regression model")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred

poly_reg_2 = PolynomialFeatures(degree = 3)
x_poly_2 = poly_reg_2.fit_transform(x)

poly_reg_2.fit(x_poly_2, y)
lin_reg_3 = LinearRegression()
lin_reg_3.fit(x_poly_2,y)

plt.scatter(x,y, color ="red")
plt.plot(x, lin_reg_3.predict(x_poly_2), color ='blue')
plt.title("poly regression model degree 3")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred_2 = lin_reg_3.predict(poly_reg_2.fit_transform([[6.5]]))
poly_model_pred_2

poly_reg_3 = PolynomialFeatures(degree = 4)
x_poly_3 = poly_reg_3.fit_transform(x)

poly_reg_3.fit(x_poly_3, y)
lin_reg_4 = LinearRegression()
lin_reg_4.fit(x_poly_3,y)

plt.scatter(x,y, color ="red")
plt.plot(x, lin_reg_4.predict(x_poly_3), color ='blue')
plt.title("poly regression model degree 4")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred_3 = lin_reg_4.predict(poly_reg_3.fit_transform([[6.5]]))
poly_model_pred_3

poly_reg_4 = PolynomialFeatures(degree = 5)
x_poly_4 = poly_reg_4.fit_transform(x)

poly_reg_4.fit(x_poly_4, y)
lin_reg_5 = LinearRegression()
lin_reg_5.fit(x_poly_4,y)

plt.scatter(x,y, color ="red")
plt.plot(x, lin_reg_5.predict(x_poly_4), color ='blue')
plt.title("poly regression model degree 5")
plt.xlabel("position level")
plt.ylabel("salary")
plt.show()

poly_model_pred_4 = lin_reg_5.predict(poly_reg_4.fit_transform([[6.5]]))
poly_model_pred_4

#svr model

from sklearn.svm import SVR
svr_model = SVR(kernel = "poly", degree=5)
svr_model.fit(x,y)

svr_model_pred = svr_model.predict([[6.5]])
svr_model_pred


#KNN MODEL

from sklearn.neighbors import KNeighborsRegressor
knn_model = KNeighborsRegressor()
knn_model.fit(x,y)

knn_model_pred = knn_model.predict([[6.5]])
knn_model_pred




















