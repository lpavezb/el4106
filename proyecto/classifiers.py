import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer


def perceptron_classifier(x_train, y_train, x_data, y_data):
    n_classes = len(np.unique(y_train))
    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(n_classes))
    y_train = y_train.astype(int)
    y_data = y_data.astype(int)

    y_train = label_binarizer.transform(y_train).astype(float)
    y_data = label_binarizer.transform(y_data).astype(float)

    n_input = x_train.shape[1]
    n_classes = y_train.shape[1]
    n_hidden = int((n_input + n_classes) * 2 / 3)
    print(n_classes)
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden])),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }

    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    keep_prob = tf.placeholder("float")

    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    predictions = tf.matmul(layer_1, weights['out']) + biases['out']

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    training_epochs = 1500
    display_step = 100
    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t1 = time()
        acc = []
        correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        for epoch in range(training_epochs):
            avg_cost = 0.0
            total_batch = int(len(x_train) / batch_size)
            x_batches = np.array_split(x_train, total_batch)
            y_batches = np.array_split(y_train, total_batch)
            for i in range(total_batch):
                batch_x, batch_y = x_batches[i], y_batches[i]
                _, c = sess.run([optimizer, cost],
                                feed_dict={
                                    x: batch_x,
                                    y: batch_y,
                                    keep_prob: 0.8
                                })
                avg_cost += c / total_batch
            data_acc = accuracy.eval({x: x_data, y: y_data, keep_prob: 1.0})
            acc.append(data_acc)
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))
                print("Accuracy validation:", data_acc)
        print("Optimization Finished!")
        t2 = time()
        confm = tf.confusion_matrix(tf.argmax(y, 1), tf.argmax(predictions, 1), num_classes=y_train.shape[1])
        confm_eval = confm.eval({x: x_data, y: y_data, keep_prob: 1.0})
        res = {"time": (t2 - t1), "confm": confm_eval, "accuracy": acc}
        confusion_matrix = np.array(res["confm"])
        print(100 * "-")
        confm_diagonal = np.diag(confusion_matrix)
        accuracy = confm_diagonal.sum() / confusion_matrix.sum()
        print(confusion_matrix)
        print("training time: {:.2f}".format(res["time"]))
        print("Accuracy: {:.7f}".format(accuracy))
        print("mean of confusion matrix diagonal: {:.2f}".format(np.mean(confm_diagonal)))
        print("standard deviation of confusion matrix diagonal: {:.2f}".format(np.std(confm_diagonal)))
        print(100 * "-")
        return [confusion_matrix, accuracy]


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
    print(confm)
    print("Accuracy: {:.4f}".format(accuracy))
    print("training time = {:.2f}".format(t2 - t1))
    print(100 * "-")
    print("classifier: {}".format(classifier))
    print(100 * "-")
    return [confm, accuracy]


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
    print(confm)
    print("Accuracy: {:.4f}".format(accuracy))
    print("training time = {:.2f}".format(t2 - t1))
    print(100 * "-")
    print("classifier: {}".format(classifier))
    print(100 * "-")
    return [confm, accuracy]
