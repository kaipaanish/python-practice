import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataset  = pd.read_csv(r"C:\Users\anish\Downloads\Investment.csv")

x = dataset.iloc[:,:-1]
y = dataset.iloc[:,4]


x = pd.get_dummies(x, dtype= int)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y ,test_size=0.2, random_state=0)


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)


bias = regressor.score(x_test, y_test)
print(bias)

variance = regressor.score(x_train, y_train)
print(variance)


x = np.append(arr = np.ones((50,1)).astype(int), values = x , axis = 1)

import statsmodels.api as sm

x_opt = x[:,[0,1,2,3,4,5]]
qt()
regressor_OLS.summary()

x1_opt = x[:,[0,1,2,3,5]]

regressor_OLS1 = sm.OLS(endog = y, exog = x1_opt).fit()
regressor_OLS1.summary()


x2_opt = x[:,[0,1,2,3]]

regressor_OLS2 = sm.OLS(endog = y, exog = x2_opt).fit()
regressor_OLS2.summary()


x3_opt = x[:,[0,1,3]]

regressor_OLS3 = sm.OLS(endog = y, exog = x3_opt).fit()
regressor_OLS3.summary()


x4_opt = x[:,[0,1]]

regressor_OLS4 = sm.OLS(endog = y, exog = x4_opt).fit()
regressor_OLS4.summary()

