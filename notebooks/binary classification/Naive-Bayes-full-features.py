from memory_profiler import profile
import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, f1_score

print("###########################################################")
print("The classification for data18 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data18_full_features_binary_classification.csv")
# X and y for input
X = df.drop('label', axis=1)
y = df['label']

train_accuracy_scores = []
train_precision_scores = []
train_f1_scores = []
test_accuracy_scores = []
test_precision_scores = []
test_f1_scores = []
train_time = []
test_time = []
average_time = []

@profile
def train_model(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model


# k-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X, y):
    # Get the training and testing data for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]


    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    average_time.append(end-start)


    y_train_pred = model.predict(X_train)


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


print("Naive Bayes Algorithm:")
print("Training Set:")
print(f"Accuracy: {train_average_accuracy:.4f}")
print(f"Precision: {train_average_precision:.4f}")
print(f"F1-Score: {train_average_f1:.4f}")
print("Test Set:")
print(f"Accuracy: {test_average_accuracy:.4f}")
print(f"Precision: {test_average_precision:.4f}")
print(f"F1-Score: {test_average_f1:.4f}")
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")



print("###########################################################")
print("The classification for data50 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data50_full_features_binary_classification.csv")
# X and y for input
X = df.drop('label', axis=1)
y = df['label']

train_accuracy_scores = []
train_precision_scores = []
train_f1_scores = []
test_accuracy_scores = []
test_precision_scores = []
test_f1_scores = []
train_time = []
test_time = []
average_time = []


# k-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X, y):
    # Get the training and testing data for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    average_time.append(end-start)

    y_train_pred = model.predict(X_train)

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


print("Naive Bayes Algorithm:")
print("Training Set:")
print(f"Accuracy: {train_average_accuracy:.4f}")
print(f"Precision: {train_average_precision:.4f}")
print(f"F1-Score: {train_average_f1:.4f}")
print("Test Set:")
print(f"Accuracy: {test_average_accuracy:.4f}")
print(f"Precision: {test_average_precision:.4f}")
print(f"F1-Score: {test_average_f1:.4f}")
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")


print("###########################################################")
print("The classification for data100 has been initiated")
print("###########################################################")

df = pd.read_csv("../../data/preprocessed/data100_full_features_binary_classification.csv")

# X and y for input
X = df.drop('label', axis=1)
y = df['label']

train_accuracy_scores = []
train_precision_scores = []
train_f1_scores = []
test_accuracy_scores = []
test_precision_scores = []
test_f1_scores = []
train_time = []
test_time = []
average_time = []


# k-fold cross-validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for train_index, test_index in kfold.split(X, y):
    # Get the training and testing data for the current fold
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    average_time.append(end-start)

    y_train_pred = model.predict(X_train)

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

print("Naive Bayes Algorithm:")
print("Training Set:")
print(f"Accuracy: {train_average_accuracy:.4f}")
print(f"Precision: {train_average_precision:.4f}")
print(f"F1-Score: {train_average_f1:.4f}")
print("Test Set:")
print(f"Accuracy: {test_average_accuracy:.4f}")
print(f"Precision: {test_average_precision:.4f}")
print(f"F1-Score: {test_average_f1:.4f}")
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")
