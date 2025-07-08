# down load dataframe from https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix,accuracy_score,classification_report
import seaborn as sns


dataset = pd.read_csv(r"C:\Users\anish\Downloads\default of credit card clients.csv", header=1)

x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


models={'Logistic Regression':LogisticRegression(),'KNN-Classifier':KNeighborsClassifier(),
        'SVM':SVC(),'GaussianNB':GaussianNB(),'BernoulliNB':BernoulliNB()}

results = []

for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred= model.predict(x_test)
    
    com=confusion_matrix(y_test, y_pred)
    acc=accuracy_score(y_test, y_pred)
    cr=classification_report(y_test, y_pred)
    bias=model.score(x_train,y_train)
    variance=model.score(x_test,y_test)
    sns.heatmap(data=com,annot=True)
    results.append({'Model':name,
                    'Confusion Matrix':com,
                    'Accuracy':acc,
                    'Classification Report':cr,
                    'Bias':bias,
                    'Variance':variance})
    plt.show()
    print(f"Model: {name}")
    print(f"Confusion Matrix:\n{com}")
    print(f"Accuracy: {acc}")
    print(f"Classification Report:\n{cr}")
    print(f"Bias: {bias}")
    print(f"Variance: {variance}\n")
    
results_df=pd.DataFrame(results)
#dataframe to csv file
results_df.to_csv('classifier_results.csv',index=False)
print("Model Evaluation Results:")
print(results_df)
    
