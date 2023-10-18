import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("mushrooms.csv")
data.head()

class_counts = data['class'].value_counts()

plt.bar('e', class_counts['e'], label='Edible')
plt.bar('p', class_counts['p'], label='Poisonous')

plt.xlabel('Class')
plt.ylabel('Count')
plt.legend()
plt.show()

X = data.drop('class', axis=1)
y = data['class']

label_encoder = LabelEncoder()
for column in X.columns:
    X[column] = label_encoder.fit_transform(X[column])

y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

logistic_classifier = LogisticRegression()
logistic_classifier.fit(X_train, y_train)

ridge_classifier = RidgeClassifier()
ridge_classifier.fit(X_train, y_train)

decision_tree_classifier = DecisionTreeClassifier()
decision_tree_classifier.fit(X_train, y_train)

naive_bayes_classifier = GaussianNB()
naive_bayes_classifier.fit(X_train, y_train)

neural_network_classifier = MLPClassifier()
neural_network_classifier.fit(X_train, y_train)

logistic_predictions = logistic_classifier.predict(X_test)
ridge_predictions = ridge_classifier.predict(X_test)
decision_tree_predictions = decision_tree_classifier.predict(X_test)
naive_bayes_predictions = naive_bayes_classifier.predict(X_test)
neural_network_predictions = neural_network_classifier.predict(X_test)

logistic_report = classification_report(y_test, logistic_predictions)
ridge_report = classification_report(y_test, ridge_predictions)
decision_tree_report = classification_report(y_test, decision_tree_predictions)
naive_bayes_report = classification_report(y_test, naive_bayes_predictions)
neural_network_report = classification_report(y_test, neural_network_predictions)

print("Logistic Regression Model Report:")
print(logistic_report)

print("Ridge Classifier Model Report:")
print(ridge_report)

print("Decision Tree Model Report:")
print(decision_tree_report)

print("Naive Bayes Model Report:")
print(naive_bayes_report)

print("Neural Network Model Report:")
print(neural_network_report)

random_forest_classifier = RandomForestClassifier()
random_forest_classifier.fit(X_train, y_train)
random_forest_predictions = random_forest_classifier.predict(X_test)

random_forest_report = classification_report(y_test, random_forest_predictions)
print("Random Forest Model Report:")
print(random_forest_report)

