import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
# Load dataset
dataset = pd.read_csv(r"C:\Users\anish\Data Science with Gen AI\Class Codes\customer_review analysis\Restaurant_Reviews.tsv", delimiter='\t', quoting=3)


# Clean the text (remove special characters, convert to lowercase)
corpus = []
for i in range(len(dataset)):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    corpus.append(review)

# TF-IDF with bigrams and stopwords removal
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2000, stop_words='english')
x = vectorizer.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Define a model evaluation function
def evaluate_model(model, name):
    model.fit(x_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(x_train))
    test_acc = accuracy_score(y_test, model.predict(x_test))
    print(f"\n--- {name} ---")
    print("Train Accuracy:", round(train_acc, 2))
    print("Test Accuracy:", round(test_acc, 2))

# Evaluate different models
evaluate_model(LogisticRegression(), "Logistic Regression")
evaluate_model(SVC(kernel='linear'), "SVM (Linear Kernel)")
evaluate_model(RandomForestClassifier(), "Random Forest")
evaluate_model(DecisionTreeClassifier(), "Decision Tree")
evaluate_model(GaussianNB(), "Naive Bayes")


param_grid = {
    'C': [0.01, 0.1, 1, 10, 100]
}
svm = SVC(kernel='linear')

# GridSearchCV to find best C value
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(x_train, y_train)

# Best model from grid search
best_svm = grid_search.best_estimator_

# Accuracy results
train_accuracy = best_svm.score(x_train, y_train)
test_accuracy = best_svm.score(x_test, y_test)

print("\n--- SVM (Linear Kernel) with Grid Search ---")
print("Best Parameters:", grid_search.best_params_)
print(f"Train Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")


# --- XGBoost ---
print("\n--- XGBoost ---")
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb.fit(x_train, y_train)
print(f"Train Accuracy: {xgb.score(x_train, y_train):.2f}")
print(f"Test Accuracy: {xgb.score(x_test, y_test):.2f}")

# --- LightGBM ---
print("\n--- LightGBM ---")
lgbm = LGBMClassifier()
lgbm.fit(x_train, y_train)
print(f"Train Accuracy: {lgbm.score(x_train, y_train):.2f}")
print(f"Test Accuracy: {lgbm.score(x_test, y_test):.2f}")