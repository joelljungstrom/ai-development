import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
import keras
import pickle
from keras import layers, models
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris, make_blobs, load_wine, load_diabetes, fetch_20newsgroups
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, VotingClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import plot_tree
from sklearn.neighbors import kneighbors_graph, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from datetime import datetime, timedelta
from ucimlrepo import fetch_ucirepo
from flask import Flask, request, jsonify, render_template


#######################################################################
'''
# 1 Linjär regression / linear regression
housing_data = fetch_california_housing()
housing_X = pd.DataFrame(data=housing_data.data, columns=housing_data.feature_names)
housing_y = pd.Series(housing_data.target)

X_train, X_test, y_train, y_test = train_test_split(housing_X, housing_y, test_size=0.25, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)

print(mse)

plt.figure(figsize=(12,6))
plt.scatter(y_pred, y_test, color='red', alpha=0.5, label='Predicted Price')
plt.scatter(y_test, y_test, color='blue', alpha=0.5, label='Actual Price') 
plt.title('Predicted vs Actual Prices')
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.legend()
plt.show()
'''

#######################################################################
'''
# 2 Logistisk regression / logistic regression
iris = load_iris()
iris_X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_y = pd.Series(data=iris.target)

# Convert the target to a DataFrame for easier manipulation
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_df['Species'] = iris_y

binary_species = iris_X[iris_y.isin([0, 1])]
binary_target = iris_y[iris_y.isin([0, 1])]

X_train, X_test, y_train, y_test = train_test_split(binary_species, binary_target, test_size=0.2, random_state=42)

model = LogisticRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)
'''
#######################################################################
'''
# 3 kNN
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

for k in range(1,4):
    knn = KNeighborsClassifier(n_neighbors=k)

    train = knn.fit(X_train, y_train)
    y_pred = train.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy (k={k}): {accuracy:.2f}')

knn = KNeighborsClassifier(n_neighbors=5)
train = knn.fit(X_train, y_train)
y_pred = train.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
'''
#######################################################################
'''
# 4 Decision Tree / Classification tree
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

print("Bästa parametrar:", grid_search.best_params_)
print("Bästa cross-validation score:", grid_search.best_score_)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nKlassificeringsrapport för bästa modell:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

importances = best_model.feature_importances_
feature_importances = pd.DataFrame(importances, index=wine.feature_names, columns=['Importance']).sort_values('Importance', ascending=False)
plt.figure(figsize=(12, 6))
feature_importances.plot(kind='bar')
plt.title('Feature Importances')
plt.ylabel('Importance')
plt.xlabel('Features')
plt.show()

plt.figure(figsize=(20,10))
plot_tree(best_model.estimators_[0], 
          feature_names=wine.feature_names, 
          class_names=wine.target_names, 
          filled=True, 
          rounded=True)
plt.title('Decision Tree from Random Forest')
plt.show()

results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 8))
plt.scatter(results['param_n_estimators'], results['mean_test_score'], c=results['param_max_depth'], cmap='viridis')
plt.colorbar(label='max_depth')
plt.xlabel('n_estimators')
plt.ylabel('Mean test score')
plt.title('GridSearchCV Results')
plt.show()

rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
print("\nKlassificeringsrapport för standardmodell:")
print(classification_report(y_test, y_pred_default, target_names=wine.target_names))
'''
#######################################################################
'''
# 5 Decision Tree / Regression tree
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
best_params = grid_search.best_params_
best_scores = grid_search.best_score_
best_model = grid_search.best_estimator_
importances = best_model.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), np.array(diabetes.feature_names)[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()

rf_y_pred = best_model.predict(X_test)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
lin_y_pred = lin_model.predict(X_test)

print(f'Random Forest MSE: {np.mean((rf_y_pred - y_test) ** 2)}')
print(f'Linear Regression MSE: {np.mean((lin_y_pred - y_test) ** 2)}')
print(f'Random Forest R²: {r2_score(y_test, rf_y_pred)}')
print(f'Linear Regression R²: {r2_score(y_test, lin_y_pred)}')
'''
#######################################################################
'''
# 6 Support Vector Machine SVM for Classification
breast_cancer_data = load_breast_cancer()
breast_cancer_X = pd.DataFrame(data=breast_cancer_data.data, columns=breast_cancer_data.feature_names)
breast_cancer_y = pd.Series(breast_cancer_data.target)

scaler = StandardScaler()
breast_cancer_X_scaled = scaler.fit_transform(breast_cancer_X)

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_X_scaled, breast_cancer_y, test_size=0.15, random_state=42)

linear_svc = svm.SVC(kernel='linear')
rbf_svc = svm.SVC(kernel='rbf')

linear_train = linear_svc.fit(X_train, y_train)
rbf_train = rbf_svc.fit(X_train, y_train)

linear_pred = linear_train.predict(X_test)
rbf_pred = rbf_train.predict(X_test)

print(f'Linear MSE: {np.mean((linear_pred - y_test) ** 2)}')
print(f'RBF MSE: {np.mean((rbf_pred - y_test) ** 2)}')

# visualiseringen är omöjlig att få rätt
x_min, x_max = breast_cancer_X_scaled[:, 0].min() - 1, breast_cancer_X_scaled[:, 0].max() + 1
y_min, y_max = breast_cancer_X_scaled[:, 1].min() - 1, breast_cancer_X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Make predictions on the mesh grid
Z = rbf_train.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plotting the decision boundary
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.5, cmap='coolwarm')

# Plot the actual data points (use only the first two features for visualization)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolors='k', marker='o', label='Training Data')
plt.scatter(X_test[:, 0], X_test[:, 1], c='red', edgecolors='k', marker='s', label='Test Data')

plt.title('Decision Boundary of SVM on Breast Cancer Dataset')
plt.xlabel(breast_cancer_X.columns[0])  # First feature
plt.ylabel(breast_cancer_X.columns[1])  # Second feature
plt.legend()
plt.show()
'''
#######################################################################
'''
# 7 Naive Bayes Text Classification
cats = ['alt.atheism', 'sci.space']
news_data = fetch_20newsgroups(subset='all', categories=cats, remove=('headers', 'footers', 'quotes'))
X, y = news_data.data, news_data.target

vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

nb = MultinomialNB()
train = nb.fit(X_train, y_train)

y_pred = train.predict(X_test)
print(classification_report(y_test, y_pred, target_names=news_data.target_names))

# Generate the confusion matrix
confusion = confusion_matrix(y_test, y_pred)
print(confusion)

correct_indices = np.where(y_pred == y_test)[0]
incorrect_indices = np.where(y_pred != y_test)[0]

# Let's display 5 correctly classified samples
print("Correctly Classified Samples:")
for i in correct_indices[:5]:
    print(f"Predicted: {news_data.target_names[y_pred[i]]}, True: {news_data.target_names[y_test[i]]}")
    print(f"Text: {news_data.data[i]}\n")

# Let's display 5 incorrectly classified samples
print("Incorrectly Classified Samples:")
for i in incorrect_indices[:5]:
    print(f"Predicted: {news_data.target_names[y_pred[i]]}, True: {news_data.target_names[y_test[i]]}")
    print(f"Text: {news_data.data[i]}\n")
'''
#######################################################################
'''
# 8 Gradient Boosting
bike_sharing = fetch_ucirepo(id=275) 
X, y = bike_sharing.data.features, bike_sharing.data.targets
X, y = pd.DataFrame(X), pd.DataFrame(y)
X = X.drop('dteday', axis=1)
X = pd.get_dummies(X, columns=['season', 'weathersit', 'yr', 'mnth', 'hr', 'weekday'], drop_first=True)

scaler = StandardScaler()
numerical_cols = ['temp', 'atemp', 'windspeed', 'hum']
X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

#mse = np.mean((y_pred - y_test) ** 2)
#print(f'Mean Squared Error: {mse}')

feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, feature_importance)
plt.xlabel('Feature Importance')
plt.title('Feature Importance of Gradient Boosting Model')
plt.show()
'''
#######################################################################
'''
# 9 Multi-layer Perceptron (MLP) for classification
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

mlp = MLPClassifier(
    hidden_layer_sizes=100,
    activation='relu',
    solver='adam'
)

train = mlp.fit(X_train, y_train)

y_pred = train.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', class_report)
'''
#######################################################################
'''
# 10 Ensemble Learning
iris = load_iris()
iris_X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_y = pd.Series(data=iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn_train = knn.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

lr = LogisticRegression()
lr_train = lr.fit(X_train, y_train)

hard_voting = VotingClassifier(estimators=[
    ('knn', knn),
    ('lr', lr),
    ('rf', rf)
], voting='hard')

hard_voting.fit(X_train, y_train)

soft_voting = VotingClassifier(estimators=[
    ('knn', knn),
    ('lr', lr),
    ('rf', rf)
], voting='soft')

soft_voting.fit(X_train, y_train)


knn_pred = knn.predict(X_test)
lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)
hard_pred = hard_voting.predict(X_test)
soft_pred = soft_voting.predict(X_test)

print(f"KNN Accuracy: {accuracy_score(y_test, knn_pred)}")
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, lr_pred)}")
print(f"Random Forest Accuracy: {accuracy_score(y_test, rf_pred)}")
print(f"Hard Voting Accuracy: {accuracy_score(y_test, hard_pred)}")
print(f"Soft Voting Accuracy: {accuracy_score(y_test, soft_pred)}")
'''
#######################################################################