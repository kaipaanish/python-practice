import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv(r"C:\Users\anish\Downloads\Salary_Data.csv")


x = dataset.iloc[:,:-1]   

y = dataset.iloc[:,-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x , y ,test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

comparision = pd.DataFrame({'Actual': y_test, 'Predict': y_pred})
print(comparision)

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title("salary vs experiance (test set)")
plt.xlabel("years of experiance")
plt.ylabel("salary")
plt.show()


m_slope = regressor.coef_
print(m_slope)

c_intercept = regressor.intercept_
print(c_intercept)

y_12 = m_slope* 12 + c_intercept
print(y_12)

bais = regressor.score(x_train, y_train)
print(bais)
variance = regressor.score(x_test, y_test)
print(variance)
# if low bias and high variance its over fitting
# if high bais and low variance its under fitting


dataset.mean()

dataset['Salary'].mean()
dataset['YearsExperience'].mean()


dataset['Salary'].median()
dataset['YearsExperience'].median()


dataset['Salary'].mode()
dataset['YearsExperience'].mode()

dataset.var()

dataset['Salary'].var()
dataset['YearsExperience'].var()

dataset["Salary"].std()
dataset['YearsExperience'].std()

# coefficient of variation
from scipy.stats import variation
variation(dataset.values)

#correlation

dataset.corr()


dataset['Salary'].corr(dataset['YearsExperience'])


#skew

dataset.skew()
dataset['Salary'].skew()


#Standard error

dataset.sem()
dataset["Salary"].sem()


#Z-score

import scipy.stats as stats

dataset.apply(stats.zscore)


#degree of freedom

a = dataset.shape[0]
b = dataset.shape[1]


degree_of_freedom = a-b
print(degree_of_freedom)


# sum of Square regressor
y_mean = np.mean(y)
SSR = np.sum((y_pred - y_mean)**2)
print(SSR)

#SSE
y = y[0:6]
SSE = np.sum((y - y_pred)**2)
print(SSE)

#SSI
mean_total = np.mean(dataset.values)
SST = np.sum((dataset.values-mean_total)**2)
print(SST)


r_square = 1-(SSR/SST)
print(r_square)



















