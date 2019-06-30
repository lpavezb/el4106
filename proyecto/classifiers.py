import numpy as np
from time import time

from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV


def svm_classifier(x_train, y_train, x_data, y_data):
    svc = svm.LinearSVC()
    t1 = time()
    parameters = {'C': [0.1, 1, 10]}
    grid = GridSearchCV(svc, parameters, cv=5)
    grid.fit(x_train, y_train)

    classifier = grid.best_estimator_
    predictions = classifier.predict(x_data)
    confm = metrics.confusion_matrix(y_data, predictions)
    confm_diagonal = np.diag(confm)
    accuracy = confm_diagonal.sum() / confm.sum()
    t2 = time()
    print(100 * "-")
    # print(confm)
    print("Accuracy: {:.4f}".format(accuracy))
    print("training time = {:.2f}".format(t2 - t1))
    print(100 * "-")
    # print("classifier: {}".format(classifier))
    print(100 * "-")
    return [accuracy, predictions]


def mlp_classifier(x_train, y_train, x_data, y_data):
    t1 = time()

    classifier = MLPClassifier()
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_data)
    confm = metrics.confusion_matrix(y_data, predictions)
    confm_diagonal = np.diag(confm)
    accuracy = confm_diagonal.sum() / confm.sum()
    t2 = time()
    print(100 * "-")
    # print(confm)
    print("Accuracy: {:.4f}".format(accuracy))
    print("training time = {:.2f}".format(t2 - t1))
    print(100 * "-")
    # print("classifier: {}".format(classifier))
    print(100 * "-")
    return [accuracy, predictions]
