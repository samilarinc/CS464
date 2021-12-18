# -*- coding: utf-8 -*-
"""SpamDetection3.1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1-5rnUGhIs5wYJNsPxCqqe8RagLRt1sdO
"""

import pandas as pd
import numpy as np
from time import perf_counter
from google.colab import drive
drive.mount('/content/drive')

# !ls drive/MyDrive/hw1_datasets/q3

features = pd.read_csv("/content/drive/MyDrive/hw1_datasets/q3/sms_train_features.csv", index_col=0)
labels = pd.read_csv("/content/drive/MyDrive/hw1_datasets/q3/sms_train_labels.csv", index_col=0)
feature_array = features.to_numpy()
label_array = labels.to_numpy().reshape(1, -1)[0]
features

ignored_indices = np.nonzero(np.sum(feature_array, axis = 0) == 0)[0]
comp_feature_array = np.delete(feature_array, ignored_indices, 1)

prob_spam = sum(label_array)/len(label_array)
prob_ham = 1 - prob_spam
n_data = len(features)
n_vocab = features.size // n_data

spam_data = comp_feature_array[label_array == 1]
ham_data = comp_feature_array[label_array == 0]
n_spam = len(spam_data)
n_ham = len(ham_data)

n_words_in_spam = np.sum(spam_data)
n_words_in_ham = np.sum(ham_data)

spam_word_probs = sum(spam_data) / n_words_in_spam
ham_word_probs = sum(ham_data) / n_words_in_ham

def predict(test_data):
    test_arr = np.delete(test_data, ignored_indices)
    predict_spam = np.log(spam_word_probs)
    predict_spam = np.where(test_arr != 0, predict_spam * test_arr, 0)
    predict_spam = predict_spam.sum()
    predict_spam += np.log(prob_spam)

    predict_ham = np.log(ham_word_probs)
    predict_ham = np.where(test_arr != 0, predict_ham * test_arr, 0)
    predict_ham *= test_arr
    predict_ham = predict_ham.sum()
    predict_ham += np.log(prob_ham)
    return int(predict_spam > predict_ham)

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
np.seterr(all = 'ignore')
start = perf_counter()
for i in range(n_data): 
    prediction = predict(feature_array[i])
    if prediction == label_array[i]:
        if prediction == 1:
            true_pos += 1
        else:
            true_neg += 1
    else:
        if prediction == 1:
            false_pos += 1
        else:
            false_neg += 1
stop = perf_counter()
print("Trainingset time:{:.4f}".format(stop - start))

print("Confusion Matrix for training set:\n", np.array([[true_pos, false_pos],[false_neg, true_neg]]))

print("Training Accuracy: {:.2f}%".format(100 * (true_neg + true_pos) / n_data))

test_features = pd.read_csv("/content/drive/MyDrive/hw1_datasets/q3/sms_test_features.csv", index_col=0)
test_labels = pd.read_csv("/content/drive/MyDrive/hw1_datasets/q3/sms_test_labels.csv", index_col=0)
test_feature_array = test_features.to_numpy()
test_label_array = test_labels.to_numpy().reshape(1, -1)[0]

true_pos = 0
true_neg = 0
false_pos = 0
false_neg = 0
n_test_data = len(test_features)
start = perf_counter()
for i in range(n_test_data): 
    prediction = predict(test_feature_array[i])
    if prediction == test_label_array[i]:
        if prediction == 1:
            true_pos += 1
        else:
            true_neg += 1
    else:
        if prediction == 1:
            false_pos += 1
        else:
            false_neg += 1
stop = perf_counter()
print("Testset Time: {:.4f}s".format(stop - start))

print("Confusion Matrix for Test Set:\n", np.array([[true_pos, false_pos], [false_neg, true_neg]]))

print("Test Accuracy:{:.2f}%".format(100 * (true_pos + true_neg) / n_test_data))

print()
