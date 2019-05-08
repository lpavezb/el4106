#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn import svm
from sklearn.metrics import confusion_matrix


def load_file():
    file = open("Restaurant_Reviews_pl.tsv")
    ar = []
    file.readline()  # read 'Review Liked' line
    for line in file:
        text, label = line.split('\t')
        ar.append([text, int(label)])
    n_ar = np.array(ar)
    return n_ar


def split_data(data):
    good = np.array(range(2))  # this is needed for vstack function
    bad = np.array(range(2))
    for x in data:
        if x[1] == 1:
            good = np.vstack((good, x))
        else:
            bad = np.vstack((bad, x))

    good = good[1:]  # delete the first row (range(2))
    bad = bad[1:]

    train_good_len = len(good) * 60 // 100
    valid_good_len = len(good) * 80 // 100
    train_bad_len = len(bad) * 60 // 100
    valid_bad_len = len(bad) * 80 // 100

    # separate between train, validation and test set for each class (60-20-20)
    train_good = good[:train_good_len]
    valid_good = good[train_good_len:valid_good_len]
    test_good = good[valid_good_len:]

    train_bad = bad[:train_bad_len]
    valid_bad = bad[train_bad_len:valid_bad_len]
    test_bad = bad[valid_bad_len:]

    train_set = np.vstack((train_good, train_bad))  # combine the sets of each train set into one train set
    valid_set = np.vstack((valid_good, valid_bad))  # combine the sets of each val set into one val set
    test_set = np.vstack((test_good, test_bad))  # combine the sets of each test set into one test set

    np.random.shuffle(train_set)
    np.random.shuffle(valid_set)
    np.random.shuffle(test_set)

    return train_set, valid_set, test_set


def split_text_and_labels(data):
    return data[:, 0], data[:, 1]


if __name__ == "__main__":
    data = load_file()
    vectorizer = CountVectorizer()
    bow = vectorizer.fit_transform(data[:, 0])

    # features = np.array(bow.toarray())
    # labels = np.array(data[:, 1])
    # train_set, valid_set, test_set = split_data(data)


