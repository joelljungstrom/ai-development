import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

#######################################################################
'''
# 1.1
housing_data = fetch_california_housing()
housing_X = pd.DataFrame(data=housing_data.data, columns=housing_data.feature_names)
housing_y = pd.Series(housing_data.target)

# 1.2
X_train, X_test, y_train, y_test = train_test_split(housing_X, housing_y, test_size=0.25, random_state=42)

# 1.3
model = LinearRegression()
model.fit(X_train, y_train)

# 1.4
y_pred = model.predict(X_test)
mse = np.mean((y_pred - y_test) ** 2)
'''
#######################################################################
'''
# 2.1
breast_cancer_data = load_breast_cancer()
breast_cancer_X = pd.DataFrame(data=breast_cancer_data.data, columns=breast_cancer_data.feature_names)
breast_cancer_y = pd.Series(breast_cancer_data.target)

# 2.2
scaler = StandardScaler()
#print(breast_cancer_X.head())
breast_cancer_X_scaled = scaler.fit_transform(breast_cancer_X)
breast_cancer_X_scaled = pd.DataFrame(breast_cancer_X_scaled, columns=breast_cancer_X.columns)
#print(breast_cancer_X_scaled.head())

X_train, X_test, y_train, y_test = train_test_split(breast_cancer_X_scaled, breast_cancer_y, test_size=0.15, random_state=42)

# 2.3 
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

# 2.4
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x=X_train, 
    y=y_train
)

# 2.5
evaluation_metrics = model.evaluate(
    x=X_test,
    y=y_test,
    return_dict=True
)

accuracy = evaluation_metrics['accuracy']
print(f'Model Accuracy: {accuracy}')
'''
#######################################################################
'''
# 3.1
iris = load_iris()
iris_X = pd.DataFrame(data=iris.data, columns=iris.feature_names)
iris_y = pd.Series(data=iris.target)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.15, random_state=42)
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# 3.2
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3, activation = 'softmax')
    ]
)

# 3.3
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 3.4
train_model = model.fit(
    x=X_train, 
    y=y_train,
    epochs=50,
    validation_split=0.2,
    verbose=0
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)

print(f'Test Accuracy: {test_acc}')
print(f'Test Accuracy: {test_loss}')

plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
plt.plot(train_model.history['accuracy'], label='Training Accuracy')
plt.plot(train_model.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_model.history['loss'], label='Training Loss')
plt.plot(train_model.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# 3.5
evaluation_metrics = model.evaluate(
    x=X_test,
    y=y_test,
    return_dict=True
)

iris_predictions = model.predict(X_test)
predicted_classes = np.argmax(iris_predictions, axis=1)
actual_classes = np.argmax(y_test, axis=1)

conf_matrix = confusion_matrix(actual_classes, predicted_classes)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.unique(actual_classes))

plt.figure(figsize=(10, 7))
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
'''
#######################################################################
'''
# 4.1
mnist = keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# 4.2
model = keras.models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# 4.3
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x=X_train, 
    y=y_train, 
    epochs=5, 
    batch_size=64, 
    validation_split=0.1
)

# 4.4
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_accuracy}, Test loss: {test_loss}')

mnist_prediction = model.predict(x=X_test)
predicted_classes = np.argmax(mnist_prediction, axis=1)
#actual_classes = np.argmax(y_test, axis=1)

incorrect_classifications = np.where(predicted_classes != y_test)[0]
incorrect_images = X_test[incorrect_classifications]
incorrect_labels = y_test[incorrect_classifications]
predicted_labels = predicted_classes[incorrect_classifications]

num_incorrect = len(incorrect_classifications)

plt.figure(figsize=(10, 5))
for i in range(min(5, num_incorrect)):
    plt.subplot(1, 5, i + 1)
    plt.imshow(incorrect_images[i].reshape(28, 28), cmap='gray')  # Reshape for display
    plt.title(f'True: {incorrect_labels[i]}\nPred: {predicted_labels[i]}')
    plt.axis('off')
plt.show()
'''
#######################################################################
# 5.1
