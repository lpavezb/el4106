# -*- coding: utf-8 -*-
"""prueba_tf_3_v07_alumnos.ipynb

Automatically generated by Colaboratory.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams
from sklearn.preprocessing import LabelBinarizer
import time


def split_data(data, train_size=0.8):
    size = len(data[0])  # number of characteristics + class number
    n_classes = int(max(data[:, size-1]))  # number of classes
    classes = [np.zeros(size)]*n_classes  # each class start with a array of 0s (for vstack function)
    for x in data:
        classes[int(x[size-1])-1] = np.vstack((classes[int(x[size-1])-1], x))  # example: if x belong to class 1, x is stored in classes[0]

    train = [np.zeros(size)]*n_classes  # the set start with a array of 0s (for vstack function)
    validation = [np.zeros(size)]*n_classes  # the set start with a array of 0s (for vstack function)
    for i in range(n_classes):
        classes[i] = classes[i][1:]  # delete the first row of each class matrix (the 0s row for vstack)
        np.random.shuffle(classes[i])  # shuffle data
        train_len = int(len(classes[i]) * train_size)  # calculate size of train set
        train = np.vstack((train, np.array(classes[i][:train_len])))  # append a new class matrix to the train matrix
        validation = np.vstack((validation, np.array(classes[i][train_len:])))  # append a new class matrix to the validation matrix

    train = train[1:]  # delete first row
    validation = validation[1:]  # delete first row
    np.random.shuffle(train)  # the data is ordered by class
    np.random.shuffle(validation)
    return train, validation


def multilayer_perceptron(x, weights, biases, keep_prob, sigmoid=1):
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    if sigmoid:
        layer_1 = tf.nn.sigmoid(layer_1)
    else:
        layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, keep_prob)
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer


def train_perceptron(x, y, predictions, cost, optimizer, x_train, y_train, x_data, y_data, n_hidden, keep_prob):
    training_epochs = 1500
    display_step = 100
    batch_size = 32

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        t1 = time.time()
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
                print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
                print("Accuracy validation:", data_acc)
        t2 = time.time()
        print("Confusion matrix validation")
        print("Optimization Finished!")
        plt.plot(range(training_epochs), acc)
        print("Accuracy validation:", accuracy.eval({x: x_data, y: y_data, keep_prob: 1.0}))
        print("----------------------------------------------")
        confm = tf.confusion_matrix(tf.argmax(y,1),tf.argmax(predictions, 1), num_classes=y_train.shape[1])
        confm_eval = confm.eval({x: x_data, y: y_data, keep_prob: 1.0})
        confm_diagonal = np.diag(confm_eval)
        print(confm_eval)
        print("training time: {:.2f}".format(t2-t1))
        print("mean of confusion matrix diagonal: {:.2f}".format(np.mean(confm_diagonal)))
        print("standard deviation of confusion matrix diagonal: {:.2f}".format(np.std(confm_diagonal)))
        print("----------------------------------------------")
        return


def separate_characteristics_and_class(data, nc, label_binarizer):
    data_x = data[:, :nc]  # delete last column (class number column)
    data_y = data[:, nc] - 1  # class column
    data_y = data_y.astype(int)  # class column, each value as int

    data_y = label_binarizer.transform(data_y).astype(float)  # one-hot encoding representation

    return data_x, data_y


def create_and_train_perceptron(x_train, y_train, x_data, y_data, n_hidden, sigmoid=1):
    n_input = x_train.shape[1]
    n_classes = y_train.shape[1]

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

    predictions = multilayer_perceptron(x, weights, biases, keep_prob, sigmoid)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

    train_perceptron(x, y, predictions, cost, optimizer, x_train, y_train, x_data, y_data, n_hidden, keep_prob)
    return


def tarea2(seed, sigmoid=1):
    random_state = seed
    np.random.seed(random_state)
    tf.set_random_seed(random_state)

    D = np.loadtxt('sensorless_tarea2_train.txt', delimiter=',')  # load train and validation data
    T = np.loadtxt('sensorless_tarea2_test.txt', delimiter=',')  # load test data

    train, validation = split_data(D)  # split train and validation sets
    nc = D.shape[1] - 1

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(int(max(D[:, nc]))))
    x_train, y_train = separate_characteristics_and_class(train, nc, label_binarizer)
    x_valid, y_valid = separate_characteristics_and_class(validation, nc, label_binarizer)
    x_test, y_test = separate_characteristics_and_class(T, nc, label_binarizer)

    n_input = x_train.shape[1]
    n_classes = y_train.shape[1]
    n_hidden = int((n_input + n_classes)*2/3)

    create_and_train_perceptron(x_train, y_train, x_test, y_test, n_hidden, sigmoid)
    return


if __name__ == '__main__':
    random = [20, 40, 60]
    plt.figure(1)
    for seed in random:
        tarea2(seed, sigmoid=1)
    plt.title("accuracy curve, sigmoid")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")

    plt.figure(2)
    for seed in random:
        tarea2(seed, sigmoid=0)
    plt.title("accuracy curve, ReLU")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")

    plt.show()
