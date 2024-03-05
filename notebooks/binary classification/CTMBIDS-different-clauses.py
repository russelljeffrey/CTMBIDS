import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from sklearn.model_selection import train_test_split 
from pyTsetlinMachine.tools import Binarizer
import pandas as pd

print("data 18 full function")
def data18_full():
    df = pd.read_csv("../../data/preprocessed/data18_full_features_binary_classification.csv")
    df = df.sample(frac=0.2)
    df.reset_index(drop=True, inplace=True)
    X = df.drop("label", axis=1)
    y = df["label"]
    b = Binarizer(max_bits_per_feature=10)
    b.fit(X.values)
    X = b.transform(X.values)
    X = X.reshape(X.shape[0], 25, 9)
    return (X, y)

print("kddcup sub function")
def kddcup_sub():
    while True:
        df = pd.read_csv("../../data/preprocessed/KDDCup99-preprocessed-sub-features.csv")
        df = df.sample(frac=0.05)
        df.reset_index(drop=True, inplace=True)
        X = df.drop("target", axis=1)
        y = df["target"]
        b = Binarizer(max_bits_per_feature=10)
        b.fit(X.values)
        X = b.transform(X.values)
        print(X.shape[1])
        if X.shape[1] == 195:
            break
    X = X.reshape(X.shape[0], 39, 5)
    return (X, y)

print("kddcup full function")
def kddcup_full():
    while True:
        df = pd.read_csv("../../data/preprocessed/KDDCup99-preprocessed-full-features.csv")
        df = df.sample(frac=0.032)
        df.reset_index(drop=True, inplace=True)
        X = df.drop("target", axis=1)
        y = df["target"]
        b = Binarizer(max_bits_per_feature=10)
        b.fit(X.values)
        X = b.transform(X.values)
        print(X.shape[1])
        if X.shape[1] == 319:
            break
    X = X.reshape(X.shape[0], 11, 29)
    return (X, y)

print("data 18 sub function")
def data18_sub():
    df = pd.read_csv("../../data/preprocessed/data18_sub_features_binary_classification.csv")
    df = df.sample(frac=0.4)
    df.reset_index(drop=True, inplace=True)
    X = df.drop("label", axis=1)
    y = df["label"]
    b = Binarizer(max_bits_per_feature=10)
    b.fit(X.values)
    X = b.transform(X.values)
    X = X.reshape(X.shape[0], 21, 5)
    return (X, y)

# Convolutional Tsetlin Machine parameters
T = 20
s = 3.9
num_clauses = [10, 20, 50, 100, 200, 450, 800, 1200, 2000, 4000]


def train_model(X_train, y_train, clause):
    model = MultiClassConvolutionalTsetlinMachine2D(clause, T, s, (3, 3))
    model.fit(X_train, y_train, epochs=10)
    return model

data18_sub_accuracy = []
data18_full_accuracy = []
kddcup99_sub_accuracy = []
kddcup99_full_accuracy = []

for i, clause in enumerate(num_clauses):
    # data 18 sub
    X, y = data18_sub()
    model = train_model(X, y, clause)
    y_train_pred = model.predict(X)
    accuracy_train = accuracy_score(y, y_train_pred)
    data18_sub_accuracy.append(accuracy_train)
    print(f"data18_sub in loop {i+1} finished")

    # data 18 full
    X, y = data18_full()
    model = train_model(X, y, clause)
    y_train_pred = model.predict(X)
    accuracy_train = accuracy_score(y, y_train_pred)
    data18_full_accuracy.append(accuracy_train)
    print(f"data18_full in loop {i+1} finished")

    # kddcup sub
    X, y = kddcup_sub()
    model = train_model(X, y, clause)
    y_train_pred = model.predict(X)
    accuracy_train = accuracy_score(y, y_train_pred)
    kddcup99_sub_accuracy.append(accuracy_train)
    print(f"kddcup_sub in loop {i+1} finished")

    # kddcup full
    X, y = kddcup_full()
    model = train_model(X, y, clause)
    y_train_pred = model.predict(X)
    accuracy_train = accuracy_score(y, y_train_pred)
    kddcup99_full_accuracy.append(accuracy_train)
    print(f"kddcup_full in loop {i+1} finished")

    print(f"loop {i+1} finished")

plt.figure(figsize=(12,8))
plt.plot(num_clauses, data18_sub_accuracy, marker='o', label='data18 sub features')
plt.plot(num_clauses, data18_full_accuracy, marker='o', label='data18 full features')
plt.plot(num_clauses, kddcup99_sub_accuracy, marker='o', label='KDDCup99 sub features')
plt.plot(num_clauses, kddcup99_sub_accuracy, marker='o', label='KDDCup99 full features')
plt.xlabel('Number of Clauses')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Number of Clauses')
plt.legend()
plt.grid(True)
plt.savefig("Accuracy for different clauses")
