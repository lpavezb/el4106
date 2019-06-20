import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from time import time
from matplotlib import pyplot as plt
from sklearn import svm, metrics
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import math
from scipy.stats import kurtosis, skew, entropy


def average_entropy(data):
    return entropy(data)/len(data)


def root_mean_square(data):
    square = 0
    for i in range(len(data)):
        square += (data[i] ** 2)

    root = math.sqrt(square/len(data))
    return root


# def plot_data(subject):
#     s1 = load_data([subject], path, empty_data)
#     plt.figure()
#     plt.suptitle("subject {}".format(subject), fontsize="x-large")
#     plt.subplot(5, 1, 1)
#     plt.plot(range(len(s1["class"])), s1["class"])
#     for i in range(8):
#         plt.subplot(5, 2, i + 3)
#         c = "channel{}".format(i + 1)
#         plt.plot(range(len(s1)), s1[c])
#         plt.title(c)
#     plt.subplots_adjust(hspace=1, wspace=0.35)


def perceptron_classifier(x_train, y_train, x_data, y_data):

    label_binarizer = LabelBinarizer()
    label_binarizer.fit(range(6))
    y_train = y_train.astype(int)
    y_data = y_data.astype(int)

    y_train = label_binarizer.transform(y_train).astype(float)
    y_data = label_binarizer.transform(y_data).astype(float)

    n_input = x_train.shape[1]
    n_classes = y_train.shape[1]
    n_hidden = int((n_input + n_classes) * 2 / 3)

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
    layer_1 = tf.nn.relu(layer_1)

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
        print("----------------------------------------------")
        confm_diagonal = np.diag(confusion_matrix)
        print(confusion_matrix)
        print("training time: {:.2f}".format(res["time"]))
        print("Accuracy: {:.7f}".format(confm_diagonal.sum() / confusion_matrix.sum()))
        print("mean of confusion matrix diagonal: {:.2f}".format(np.mean(confm_diagonal)))
        print("standard deviation of confusion matrix diagonal: {:.2f}".format(np.std(confm_diagonal)))
        print("----------------------------------------------")


def svm_classifier(x_train, y_train, x_data, y_data):
    svc = svm.SVC(kernel='rbf',  gamma='auto', probability=True)
    t1 = time()
    parameters = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['poly', 'rbf', 'linear'], 'gamma': ['auto', 'scale'], 'degree': [2, 3, 4]}
    grid = GridSearchCV(svc, parameters, cv=5)
    grid.fit(x_train, y_train)

    classifier = grid.best_estimator_
    predictions = classifier.predict(x_data)
    confm = metrics.confusion_matrix(y_data, predictions)
    confm_diagonal = np.diag(confm)
    t2 = time()
    print("-----------------------------------------------")
    print(confm)
    print("classifier: {}".format(classifier))
    print("Accuracy: {:.4f}".format(confm_diagonal.sum() / confm.sum()))
    print("training time = {:.2f}".format(t2 - t1))
    print("-----------------------------------------------")


def load_data(subjects, path, data, drop7=True):
    ndata = data.copy()
    for subject in subjects:
        new_path = path + subject + "/"
        d1, d2 = os.listdir(new_path)
        ndata = ndata.append(pd.read_csv(new_path + d1, sep="\t"), sort=True)
        ndata = ndata.append(pd.read_csv(new_path + d2, sep="\t"), sort=True)
    if drop7:
        ndata = ndata[ndata["class"] != 7]
    ndata = ndata.drop("time", 1)
    return ndata.drop("class", 1), ndata["class"]


def windows_without_zeros(data, width, step):
    start = 0
    res = []
    while (start + width) < len(data):
        data_slice = data[start:start+width].reset_index()
        if (data_slice["class"][0] == data_slice["class"][199]) and data_slice["class"][0] != 0:
            res.append(data_slice)
        start += width
        start += step
    return res


def get_features(window):
    features = {}
    for i in range(1, 9):
        max_value = np.min(window["channel" + str(i)])
        min_value = np.max(window["channel" + str(i)])
        features["mean"+ str(i)] = np.mean(window["channel" + str(i)])
        features["variance" + str(i)] = np.var(window["channel" + str(i)])
        features["min" + str(i)] = min_value
        features["max" + str(i)] = max_value
        features["range" + str(i)] = max_value - min_value
        features["RMS" + str(i)] = root_mean_square(window["channel" + str(i)])
        features["skew" + str(i)] = skew(window["channel" + str(i)])
        # features["AvEnt" + str(i)] = average_entropy(window["channel" + str(i)])
        features["kurtosis" + str(i)] = kurtosis(window["channel" + str(i)])

    features["label"] = window["class"][0]
    return features


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(42)
    path = "EMG_data/"
    subjects = os.listdir(path)
    subjects.sort()

    test_len = len(subjects)-8
    train_validation = subjects[:test_len]
    t_len = int(len(train_validation)*0.6)

    test_subjects = subjects[test_len:]
    train_subjects = train_validation[:t_len]
    validation_subjects = train_validation[t_len:]

    # train_subjects = ["01", "02", "03", "04"]
    # validation_subjects = ["06", "07", "08", "09"]
    columns = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "class"]
    empty_data = pd.DataFrame(columns=columns)

    print(100 * '-')
    print(44 * '-' + "loading data" + 44 * '-')
    t = time()
    x_train, y_train = load_data(train_subjects, path, empty_data)
    x_valid, y_valid = load_data(validation_subjects, path, empty_data)
    # test = load_data(test_subjects, path, empty_data)

    print("train_samples: {}".format(len(x_train)))
    print("validation_samples: {}".format(len(x_valid)))
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(36 * '-' + "Standardization of datasets" + 37 * '-')
    t = time()
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_valid = scaler.transform(x_valid)
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(40 * '-' + "calculating windows" + 41 * '-')
    t = time()
    t_values = np.hstack((x_train, y_train.to_numpy().reshape(len(x_train), 1)))
    v_values = np.hstack((x_valid, y_valid.to_numpy().reshape(len(x_valid), 1)))
    train = pd.DataFrame(columns=columns, data=t_values)
    valid = pd.DataFrame(columns=columns, data=v_values)

    train_windows = windows_without_zeros(train, 200, 200)
    valid_windows = windows_without_zeros(valid, 200, 200)

    print("train_windows: {}".format(len(train_windows)))
    print("validation_windows: {}".format(len(valid_windows)))
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(40 * '-' + "calculating features" + 40 * '-')
    t = time()
    # features = ["mean", "variance", "min", "max", "range", "RMS", "skew", "AvEnt", "kurtosis"]
    features = ["mean", "variance", "min", "max", "range", "RMS", "skew", "kurtosis"]
    for feature in features:
        print(feature)
    cols = []
    for i in range(1, 9):
        for feature in features:
            cols.append(feature + str(i))
    cols.append("label")

    t_windows = pd.DataFrame(columns=cols)
    for window in train_windows:
        window_features = get_features(window)
        t_windows = t_windows.append(window_features, ignore_index=True)
    v_windows = pd.DataFrame(columns=cols)
    for window in train_windows:
        window_features = get_features(window)
        v_windows = v_windows.append(window_features, ignore_index=True)
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(44 * '-' + "value counts" + 44 * '-')
    print(t_windows["label"].value_counts())
    print(100 * "-")

    print(41 * '-' + "feature selection" + 42 * '-')
    x_train, y_train = t_windows.drop("label", 1).to_numpy(), t_windows["label"].to_numpy()
    x_valid, y_valid = v_windows.drop("label", 1).to_numpy(), v_windows["label"].to_numpy()

    selector = SelectKBest(mutual_info_classif, k=20)
    selector.fit(x_train, y_train)
    train = selector.transform(x_train)
    valid = selector.transform(x_valid)
    print(100 * "-")
    svm_classifier(train, y_train, valid, y_valid)
    perceptron_classifier(train, y_train, valid, y_valid)


