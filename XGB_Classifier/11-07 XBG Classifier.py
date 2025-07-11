import numpy as np
import pandas as pd

dataset = pd.read_csv(r"C:\Users\anish\Downloads\Churn_Modelling.csv")

x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,-1].values

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

from xgboost import XGBClassifier
classifier = XGBClassifier(random_state=0)
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)


from sklearn.metrics import confusion_matrix, accuracy_score
acc = accuracy_score(y_test, y_pred)
print(f"The accuray of the model XGB classifier is {acc}")
CM = confusion_matrix(y_test, y_pred)
print(f"The confusion matrix of the above model is \n{CM}")
