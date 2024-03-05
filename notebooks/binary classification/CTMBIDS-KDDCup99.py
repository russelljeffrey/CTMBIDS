import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix
from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
from sklearn.model_selection import KFold
from memory_profiler import profile
from pyTsetlinMachine.tools import Binarizer
import time
import seaborn as sns
import pandas as pd

while True:
    df = pd.read_csv("../../data/preprocessed/KDDCup99-preprocessed-sub-features.csv")
    shuffled_df = df.sample(frac=1)
    label_0_rows = shuffled_df[shuffled_df['target'] == 0].sample(n=50000)
    label_1_rows = shuffled_df[shuffled_df['target'] == 1].sample(n=50000)
    selected_rows = pd.concat([label_0_rows, label_1_rows])
    df = selected_rows.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    X = df.drop("target", axis=1)
    y = df["target"]
    b = Binarizer(max_bits_per_feature=10)
    b.fit(X.values)
    X = b.transform(X.values)
    print(X.shape)
    if X.shape[1] == 208:
        break

X = X.reshape(X.shape[0], 16, 13)

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
confusion_matrices = []
train_time = []
test_time = []
average_time = []

train_losses = []
val_losses = []


@profile
def train_model(X_train, y_train):
    model = MultiClassConvolutionalTsetlinMachine2D(num_clauses, T, s, (3, 3))
    model.fit(X_train, y_train, epochs= 10)
    return model

print("###########################################################")
print("The classification for KDDCup99 sub features has been initiated")
print("###########################################################")


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    train_time.append(end - start)

    y_train_pred = model.predict(X_train)
    train_loss = 1.0 - accuracy_score(y_train, y_train_pred)
    train_losses.append(train_loss)

    y_val_pred = model.predict(X_test)
    val_loss = 1.0 - accuracy_score(y_test, y_val_pred)
    val_losses.append(val_loss)

    y_train_pred = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    start = time.time()
    y_test_pred = model.predict(X_test)
    end = time.time()
    test_time.append(end - start)

    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)

    confusion_matrix_binary = confusion_matrix(y_test, y_test_pred)

    accuracy_scores_train.append(accuracy_train)
    precision_scores_train.append(precision_train)
    f1_scores_train.append(f1_train)
    accuracy_scores_test.append(accuracy_test)
    precision_scores_test.append(precision_test)
    f1_scores_test.append(f1_test)
    confusion_matrices.append(confusion_matrix_binary)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 10 + 1), train_losses, label='Training Loss') # 10 is the number of folds
plt.plot(range(1, 10 + 1), val_losses, label='Validation Loss') # 10 is the number of folds
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Train-Validation-loss-plot - CTMBIDS-KDDCup99-sub.png")

average_accuracy_train = np.mean(accuracy_scores_train)
average_precision_train = np.mean(precision_scores_train)
average_f1_train = np.mean(f1_scores_train)
average_accuracy_test = np.mean(accuracy_scores_test)
average_precision_test = np.mean(precision_scores_test)
average_f1_test = np.mean(f1_scores_test)
average_confusion_matrix = np.mean(confusion_matrices, axis=0)
average_time.append(np.mean(train_time))
average_time.append(np.mean(test_time))

print("Convolutional Tsetlin Machine - 10-fold Cross Validation Results")
print("Training Set:")
print("Accuracy: {:.4f}".format(average_accuracy_train))
print("Precision: {:.4f}".format(average_precision_train))
print("F1-score: {:.4f}".format(average_f1_train))
print("Test Set:")
print("Accuracy: {:.4f}".format(average_accuracy_test))
print("Precision: {:.4f}".format(average_precision_test))
print("F1-score: {:.4f}".format(average_f1_test))
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")

plt.figure(figsize=(8, 6))
sns.heatmap(average_confusion_matrix, annot=True, fmt=".0f", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.savefig("CTMBIDS Confusion Matrix - KDDCup99 - sub features")

while True:
    df = pd.read_csv("../../data/preprocessed/KDDCup99-preprocessed-full-features.csv")
    shuffled_df = df.sample(frac=1)
    label_0_rows = shuffled_df[shuffled_df['target'] == 0].sample(n=50000)
    label_1_rows = shuffled_df[shuffled_df['target'] == 1].sample(n=50000)
    selected_rows = pd.concat([label_0_rows, label_1_rows])
    df = selected_rows.sample(frac=1)
    df.reset_index(drop=True, inplace=True)
    X = df.drop("target", axis=1)
    y = df["target"]
    b = Binarizer(max_bits_per_feature=10)
    b.fit(X.values)
    X = b.transform(X.values)
    print(X.shape[1])
    if X.shape[1] == 354:
        break
X = X.reshape(X.shape[0], 59, 6)


# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True)
accuracy_scores_train = []
precision_scores_train = []
f1_scores_train = []
accuracy_scores_test = []
precision_scores_test = []
f1_scores_test = []
confusion_matrices = []
train_time = []
test_time = []
average_time = []

train_losses = []
val_losses = []


print("###########################################################")
print("The classification for KDDCup99 full features has been initiated")
print("###########################################################")


for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



    start = time.time()
    model = train_model(X_train, y_train)
    end = time.time()
    train_time.append(end - start)

    y_train_pred = model.predict(X_train)
    train_loss = 1.0 - accuracy_score(y_train, y_train_pred)
    train_losses.append(train_loss)

    y_val_pred = model.predict(X_test)
    val_loss = 1.0 - accuracy_score(y_test, y_val_pred)
    val_losses.append(val_loss)

    y_train_pred = model.predict(X_train)
    accuracy_train = accuracy_score(y_train, y_train_pred)
    precision_train = precision_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)

    start = time.time()
    y_test_pred = model.predict(X_test)
    end = time.time()
    test_time.append(end - start)

    accuracy_test = accuracy_score(y_test, y_test_pred)
    precision_test = precision_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)

    confusion_matrix_binary = confusion_matrix(y_test, y_test_pred)

    accuracy_scores_train.append(accuracy_train)
    precision_scores_train.append(precision_train)
    f1_scores_train.append(f1_train)
    accuracy_scores_test.append(accuracy_test)
    precision_scores_test.append(precision_test)
    f1_scores_test.append(f1_test)
    confusion_matrices.append(confusion_matrix_binary)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 10 + 1), train_losses, label='Training Loss') # 10 is the number of folds
plt.plot(range(1, 10 + 1), val_losses, label='Validation Loss') # 10 is the number of folds
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("Train-Validation-loss-plot - CTMBIDS-KDDCup99-full.png")

average_accuracy_train = np.mean(accuracy_scores_train)
average_precision_train = np.mean(precision_scores_train)
average_f1_train = np.mean(f1_scores_train)
average_accuracy_test = np.mean(accuracy_scores_test)
average_precision_test = np.mean(precision_scores_test)
average_f1_test = np.mean(f1_scores_test)
average_confusion_matrix = np.mean(confusion_matrices, axis=0)
average_time.append(np.mean(train_time))
average_time.append(np.mean(test_time))

print("Convolutional Tsetlin Machine - 10-fold Cross Validation Results")
print("Training Set:")
print("Accuracy: {:.4f}".format(average_accuracy_train))
print("Precision: {:.4f}".format(average_precision_train))
print("F1-score: {:.4f}".format(average_f1_train))
print("Test Set:")
print("Accuracy: {:.4f}".format(average_accuracy_test))
print("Precision: {:.4f}".format(average_precision_test))
print("F1-score: {:.4f}".format(average_f1_test))
print(f"Average Time: {average_time[0]:.4f} (Training), {average_time[1]:.4f} (Testing)")

plt.figure(figsize=(8, 6))
sns.heatmap(average_confusion_matrix, annot=True, fmt=".0f", cmap="Blues")
plt.xlabel("Predicted Class")
plt.ylabel("True Class")
plt.title("Confusion Matrix")
plt.savefig("CTMBIDS Confusion Matrix - KDDCup99 - full features")