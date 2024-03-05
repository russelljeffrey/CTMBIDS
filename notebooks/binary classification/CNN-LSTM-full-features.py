import pandas as pd
import numpy as np
import time

from tensorflow.keras.models import Model
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, f1_score
from tensorflow.keras.layers import Dense, LSTM, MaxPooling1D, Conv1D, Conv1D, MaxPooling1D, LSTM, Dense
from tensorflow.keras.models import Model
from memory_profiler import profile
from tensorflow.keras.models import Model

SAMPLES = 100000

train_accuracy_scores = []
train_precision_scores = []
train_f1_scores = []
test_accuracy_scores = []
test_precision_scores = []
test_f1_scores = []
train_time = []
test_time = []
average_time = []

class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()

        self.conv1d = Conv1D(128, kernel_size=3, activation='relu', padding='same')
        self.max_pooling1d = MaxPooling1D(pool_size=2)
        self.lstm = LSTM(64)
        self.dense = Dense(1, activation='sigmoid')

    def call(self, inputs):
        x = self.conv1d(inputs)
        x = self.max_pooling1d(x)
        x = self.lstm(x)
        x = self.dense(x)
        return x


@profile
def train_and_test(X, y):
    # stratified k-fold cross-validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_index, test_index in kfold.split(X, y):

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = MyModel()

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        start = time.time()
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        end = time.time()
        train_time.append(end - start)


        y_train_pred = model.predict(X_train)
        y_train_pred = np.round(y_train_pred).flatten()

        train_accuracy = accuracy_score(y_train, y_train_pred)
        train_precision = precision_score(y_train, y_train_pred)
        train_f1 = f1_score(y_train, y_train_pred)
        train_accuracy_scores.append(train_accuracy)
        train_precision_scores.append(train_precision)
        train_f1_scores.append(train_f1)

        start = time.time()
        y_test_pred = model.predict(X_test)
        end = time.time()
        test_time.append(end-start)
        y_test_pred = np.round(y_test_pred).flatten()

        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred)
        test_f1 = f1_score(y_test, y_test_pred)
        test_accuracy_scores.append(test_accuracy)
        test_precision_scores.append(test_precision)
        test_f1_scores.append(test_f1)

    train_average_accuracy = np.mean(train_accuracy_scores)
    train_average_precision = np.mean(train_precision_scores)
    train_average_f1 = np.mean(train_f1_scores)
    test_average_accuracy = np.mean(test_accuracy_scores)
    test_average_precision = np.mean(test_precision_scores)
    test_average_f1 = np.mean(test_f1_scores)
    average_time.append(np.mean(train_time))
    average_time.append(np.mean(test_time))

    print("Training Average Accuracy:", train_average_accuracy)
    print("Training Average Precision:", train_average_precision)
    print("Training Average F1 Score:", train_average_f1)
    print("Testing Average Accuracy:", test_average_accuracy)
    print("Testing Average Precision:", test_average_precision)
    print("Testing Average F1 Score:", test_average_f1)
    print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")


print("###########################################################")
print("The classification for data18 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data18_full_features_binary_classification.csv")
df = df.sample(n = SAMPLES)

timesteps = df.shape[1]
features = 1

df.reset_index(drop=True, inplace=True)
X = df.values.reshape((-1, timesteps, features))
y = df["label"]

train_and_test(X, y)

print("###########################################################")
print("The classification for data50 has been initiated")
print("###########################################################")


df = pd.read_csv("../../data/preprocessed/data50_full_features_binary_classification.csv")
df = df.sample(n = SAMPLES)

timesteps = df.shape[1]
features = 1

df.reset_index(drop=True, inplace=True)
X = df.values.reshape((-1, timesteps, features))
y = df["label"]

train_and_test(X, y)

print("###########################################################")
print("The classification for data100 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data100_full_features_binary_classification.csv")
df = df.sample(n = SAMPLES)

timesteps = df.shape[1]
features = 1

df.reset_index(drop=True, inplace=True)
X = df.values.reshape((-1, timesteps, features))
y = df["label"]

train_and_test(X, y)
