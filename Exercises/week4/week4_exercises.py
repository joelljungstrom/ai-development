import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers, models
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, root_mean_squared_error, r2_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_iris, make_blobs, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
from datetime import datetime, timedelta

#######################################################################
'''
# 1.1 Linjär regression / linear regression
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
# 2.1 Binär klassificering / binary classification
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
# 3.1 Flerklass-klassificering / Multiclass classification
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
# 4.1 CNN Neural Network
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
'''
# 5.1 Textklassificering / text classification
imdb = keras.datasets.imdb
max_features = 20000
maxlen = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)


# 5.2
X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

print(X_train)

# 5.3
inputs = keras.Input(shape=(None,), dtype="int32")
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Embedding(input_dim=max_features, output_dim=32, input_length=maxlen),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(32),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ]
)

# 5.4
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

train = model.fit(
    x=X_train,
    y=y_train,
    epochs=5, 
    batch_size=64, 
    validation_split=0.1
)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print(f'Test accuracy: {test_acc}, test loss: {test_loss}')

# 5.5
custom_reviews = [
    'The Dark Knight is like a gourmet burger with a side of existential dread—Heath Ledger’s Joker had me laughing and questioning my life choices all at once. Just remember to keep the lights on; this film might make you scared of the dark!',
    'In The Dark Knight, Batman sounds like he’s gargling gravel while Ledger’s Joker steals the show with a performance so good it should come with a warning label. Grab your popcorn and prepare to question everything you thought you knew about good and evil!',
    'Titanic is three hours of melodrama and frozen romance that could have been a TikTok video. If I wanted to watch people freeze to death while cuddling on a door, I’d just watch my friends during a Swedish winter!'
]

word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in text])

def classify_review(review):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts([review])
    sequence = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(sequence, maxlen=maxlen)
    prediction = model.predict(padded)[0][0]
    return "Positive" if prediction > 0.5 else "Negative"

for review in custom_reviews:
    print(f"Review: {review}")
    print(f"Sentiment: {classify_review(review)}\n")
'''
#######################################################################

# 6.1 Klustering / Clustering with kNN
'''
X, y = make_blobs(
    n_samples=500, 
    n_features=2, 
    centers=4,
    cluster_std=0.6, 
    random_state=0
)
#print(X)
#print(y)
plt.scatter(X[:, 0], X[:, 1], s=30)
plt.title('Generated Blobs')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = KMeans(
    n_clusters=4,
    max_iter=300,
    verbose=0
)

train = model.fit(X)

centers = model.cluster_centers_
labels = model.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels, s=30, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')  # Cluster centers
plt.title('KMeans Clustering Results')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
'''
#######################################################################
'''
# 7.1 Dimensionalitetsreduction / Dimension reduction
fashion_mnist = keras.datasets.fashion_mnist
(X_train, _), (X_test, _) = fashion_mnist.load_data()

X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

X_train = X_train.reshape((len(X_train)), 28*28)
X_test = X_test.reshape((len(X_test)), 28*28)

input_dim = X_train.shape[1]
encoding_dim = 32
input_layer = layers.Input(shape=(input_dim,))

encoded = layers.Dense(encoding_dim, activation='relu')(input_layer)

decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = keras.models.Model(input_layer, decoded)

encoder = keras.models.Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the autoencoder
autoencoder.fit(
    X_train, X_train,
    epochs=50, 
    batch_size=256,
    shuffle=True,
    validation_data=(X_test, X_test)
)

encoded_imgs = encoder.predict(X_test)
decoded_imgs = autoencoder.predict(X_test)


n = 10
plt.figure(figsize=(20,4))
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
    plt.axis('off')
plt.show()
'''
#######################################################################
'''
# 8 Tidsserieprognos / Time-series analysis
airtravel = pd.read_csv('ai-development/Exercises/week4/AirPassengers.csv',index_col=0)
airtravel = airtravel.dropna()

scaler = MinMaxScaler(feature_range=(0,1))
scaled = airtravel.copy()
scaled['#Passengers'] = scaler.fit_transform(airtravel[['#Passengers']])

def windowing(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step):
        X.append(data.iloc[i:i + time_step, 0].values) 
        Y.append(data.iloc[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 10
X, y = windowing(scaled, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(time_step, 1)),
        tf.keras.layers.Dense(1)
    ]
)

model.compile(optimizer='adam', loss='mse')

train = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0)

train_predict = model.predict(X_train).flatten()
test_predict = model.predict(X_test).flatten()

train_predict = scaler.inverse_transform(train_predict.reshape(-1, 1))
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
test_predict = scaler.inverse_transform(test_predict.reshape(-1, 1))
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12, 6))
plt.plot(airtravel.index[time_step:], scaler.inverse_transform(scaled[time_step:]), label='Actual')
#plt.plot(airtravel.index[time_step:train_size], train_predict, label='Train Prediction')
#plt.plot(airtravel.index[train_size + time_step: train_size + time_step + len(test_predict)], test_predict, label='Test Prediction')  # Correct the index range for test predictions
plt.legend()
plt.title('Time Series Forecasting of Air Passengers')
plt.show()
'''
#######################################################################
'''
# 9 Överföringslärande / Transfer learning
flowers_train = 'ai-development/Exercises/week4/flowers/train'
flowers_test = 'ai-development/Exercises/week4/flowers/test'

base_model = keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False 

model = tf.keras.models.Sequential(
    [
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(5, activation='softmax')
    ]
)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'flowers_train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'flowers_test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=20
)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()

# Finjustera modellen
for layer in base_model.layers[-4:]:
    layer.trainable = True

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Träna modellen igen med finjustering
history_fine = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32,
    epochs=10
)
'''
#######################################################################

# 10 Hyperparameterinställning / Hyper parameter tuning

# Ladda dataset
wine = load_wine()
X, y = wine.data, wine.target

# Dela upp data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definiera parameterrutnät
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Skapa en RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Utför GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Skriv ut bästa parametrar och score
print("Bästa parametrar:", grid_search.best_params_)
print("Bästa cross-validation score:", grid_search.best_score_)

# Utvärdera den bästa modellen på testdata
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("\nKlassificeringsrapport för bästa modell:")
print(classification_report(y_test, y_pred, target_names=wine.target_names))

# Visualisera resultat
results = pd.DataFrame(grid_search.cv_results_)
plt.figure(figsize=(12, 8))
plt.scatter(results['param_n_estimators'], results['mean_test_score'], c=results['param_max_depth'], cmap='viridis')
plt.colorbar(label='max_depth')
plt.xlabel('n_estimators')
plt.ylabel('Mean test score')
plt.title('GridSearchCV Results')
plt.show()

# Träna en modell med standardparametrar för jämförelse
rf_default = RandomForestClassifier(random_state=42)
rf_default.fit(X_train, y_train)
y_pred_default = rf_default.predict(X_test)
print("\nKlassificeringsrapport för standardmodell:")
print(classification_report(y_test, y_pred_default, target_names=wine.target_names))

