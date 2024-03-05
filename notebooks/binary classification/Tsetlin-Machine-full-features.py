import numpy as np
from sklearn.metrics import accuracy_score, precision_score, f1_score
from pyTsetlinMachine.tm import MultiClassTsetlinMachine
from sklearn.model_selection import KFold
import pandas as pd
from pyTsetlinMachine.tools import Binarizer
from memory_profiler import profile
import time

SAMPLES = 100000

print("###########################################################")
print("The classification for data18 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data18_full_features_binary_classification.csv")
df = df.sample(n = SAMPLES)
df.reset_index(drop=True, inplace=True)
X = df.drop("label", axis=1)
y = df["label"]

b = Binarizer(max_bits_per_feature=10)
b.fit(X.values)
X = b.transform(X.values)

# Tsetlin Machine hyperparameters
num_clauses = 2000
T = 20
s = 3.9

# Perform 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)
accuracy_scores_train = []
precision_scores_train = []
f1_scores_train = []
accuracy_scores_test = []
precision_scores_test = []
f1_scores_test = []
train_time = []
test_time = []
average_time = []

@profile
def train_model(X_train, y_train):
    model = MultiClassTsetlinMachine(num_clauses, T, s)
    model.fit(X_train, y_train, epochs=10)
    return model


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    train_time.append(end - start)


    y_train_pred = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    start = time.time()
    y_test_pred = model.predict(X_test)
    end = time.time()
    test_time.append(end - start)

    accuracy_scores_train.append(accuracy_train)
    precision_scores_train.append(precision_train)
    f1_scores_train.append(f1_train)
    accuracy_scores_test.append(accuracy_score(y_test, y_test_pred))
    precision_scores_test.append(precision_score(y_test, y_test_pred))
    f1_scores_test.append(f1_score(y_test, y_test_pred))

average_accuracy_train = np.mean(accuracy_scores_train)
average_precision_train = np.mean(precision_scores_train)
average_f1_train = np.mean(f1_scores_train)
average_accuracy_test = np.mean(accuracy_scores_test)
average_precision_test = np.mean(precision_scores_test)
average_f1_test = np.mean(f1_scores_test)
average_time.append(np.mean(train_time))
average_time.append(np.mean(test_time))

print("Tsetlin Machine - 10-fold Cross Validation Results")
print("Training Set:")
print("Accuracy: {:.4f}".format(average_accuracy_train))
print("Precision: {:.4f}".format(average_precision_train))
print("F1-score: {:.4f}".format(average_f1_train))
print("Test Set:")
print("Accuracy: {:.4f}".format(average_accuracy_test))
print("Precision: {:.4f}".format(average_precision_test))
print("F1-score: {:.4f}".format(average_f1_test))
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")


print("###########################################################")
print("The classification for data50 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data50_full_features_binary_classification.csv", low_memory=True)
df = df.sample(n = SAMPLES)
df.reset_index(drop=True, inplace=True)
X = df.drop("label", axis=1)
y = df["label"]

b = Binarizer(max_bits_per_feature=10)
b.fit(X.values)
X = b.transform(X.values)

num_clauses = 2000
T = 20
s = 3.9

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)
accuracy_scores_train = []
precision_scores_train = []
f1_scores_train = []
accuracy_scores_test = []
precision_scores_test = []
f1_scores_test = []
train_time = []
test_time = []
average_time = []

@profile
def train_model(X_train, y_train):
    model = MultiClassTsetlinMachine(num_clauses, T, s)
    model.fit(X_train, y_train, epochs=10)
    return model


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    train_time.append(end - start)

    y_train_pred = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    start = time.time()
    y_test_pred = model.predict(X_test)
    end = time.time()
    test_time.append(end - start)

    accuracy_scores_train.append(accuracy_train)
    precision_scores_train.append(precision_train)
    f1_scores_train.append(f1_train)
    accuracy_scores_test.append(accuracy_score(y_test, y_test_pred))
    precision_scores_test.append(precision_score(y_test, y_test_pred))
    f1_scores_test.append(f1_score(y_test, y_test_pred))

average_accuracy_train = np.mean(accuracy_scores_train)
average_precision_train = np.mean(precision_scores_train)
average_f1_train = np.mean(f1_scores_train)
average_accuracy_test = np.mean(accuracy_scores_test)
average_precision_test = np.mean(precision_scores_test)
average_f1_test = np.mean(f1_scores_test)
average_time.append(np.mean(train_time))
average_time.append(np.mean(test_time))

print("Tsetlin Machine - 10-fold Cross Validation Results")
print("Training Set:")
print("Accuracy: {:.4f}".format(average_accuracy_train))
print("Precision: {:.4f}".format(average_precision_train))
print("F1-score: {:.4f}".format(average_f1_train))
print("Test Set:")
print("Accuracy: {:.4f}".format(average_accuracy_test))
print("Precision: {:.4f}".format(average_precision_test))
print("F1-score: {:.4f}".format(average_f1_test))
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")

print("###########################################################")
print("The classification for data100 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data100_full_features_binary_classification.csv")
df = df.sample(n = SAMPLES)
df.reset_index(drop=True, inplace=True)
X = df.drop("label", axis=1)
y = df["label"]

b = Binarizer(max_bits_per_feature=10)
b.fit(X.values)
X = b.transform(X.values)

num_clauses = 2000
T = 20
s = 3.9

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)
accuracy_scores_train = []
precision_scores_train = []
f1_scores_train = []
accuracy_scores_test = []
precision_scores_test = []
f1_scores_test = []
train_time = []
test_time = []
average_time = []

@profile
def train_model(X_train, y_train):
    model = MultiClassTsetlinMachine(num_clauses, T, s)
    model.fit(X_train, y_train, epochs=10)
    return model


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    train_time.append(end - start)

    y_train_pred = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    start = time.time()
    y_test_pred = model.predict(X_test)
    end = time.time()
    test_time.append(end - start)

    accuracy_scores_train.append(accuracy_train)
    precision_scores_train.append(precision_train)
    f1_scores_train.append(f1_train)
    accuracy_scores_test.append(accuracy_score(y_test, y_test_pred))
    precision_scores_test.append(precision_score(y_test, y_test_pred))
    f1_scores_test.append(f1_score(y_test, y_test_pred))

average_accuracy_train = np.mean(accuracy_scores_train)
average_precision_train = np.mean(precision_scores_train)
average_f1_train = np.mean(f1_scores_train)
average_accuracy_test = np.mean(accuracy_scores_test)
average_precision_test = np.mean(precision_scores_test)
average_f1_test = np.mean(f1_scores_test)
average_time.append(np.mean(train_time))
average_time.append(np.mean(test_time))

print("Tsetlin Machine - 10-fold Cross Validation Results")
print("Training Set:")
print("Accuracy: {:.4f}".format(average_accuracy_train))
print("Precision: {:.4f}".format(average_precision_train))
print("F1-score: {:.4f}".format(average_f1_train))
print("Test Set:")
print("Accuracy: {:.4f}".format(average_accuracy_test))
print("Precision: {:.4f}".format(average_precision_test))
print("F1-score: {:.4f}".format(average_f1_test))
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")
