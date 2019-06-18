from time import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import metrics
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer


def plot_data(subject):
    s1 = load_data([subject], path, empty_data)
    plt.figure()
    plt.suptitle("subject {}".format(subject), fontsize="x-large")
    plt.subplot(5, 1, 1)
    plt.plot(range(len(s1["class"])), s1["class"])
    for i in range(8):
        plt.subplot(5, 2, i + 3)
        c = "channel{}".format(i + 1)
        plt.plot(range(len(s1)), s1[c])
        plt.title(c)
    plt.subplots_adjust(hspace=1, wspace=0.35)


def load_data(subjects, path, data, drop7=True):
    ndata = data.copy()
    for subject in subjects:
        new_path = path + subject + "/"
        d1, d2 = os.listdir(new_path)
        ndata = ndata.append(pd.read_csv(new_path + d1, sep="\t"))
        ndata = ndata.append(pd.read_csv(new_path + d2, sep="\t"))
    if drop7:
        ndata = ndata[ndata["class"] != 7]
    return ndata


def windows(data, width, step):
    start = 0
    res = []
    while (start + width) < len(data):
        data_slice = data[start:start+width].reset_index()
        if data_slice["class"][0] == data_slice["class"][199]:
            res.append(data_slice)
        start += width
        start += step
    return res


def get_features(window):
    features = {}
    for i in range(1, 9):
        features["mean"+ str(i)] = np.mean(window["channel" + str(i)])
        features["min" + str(i)] = np.min(window["channel" + str(i)])
        features["max" + str(i)] = np.max(window["channel" + str(i)])
    features["label"] = window["class"][0]
    return features


def perceptron_classifier(train, valid):
    x_train, y_train = train.drop("label", 1), train["label"]
    x_data, y_data = valid.drop("label", 1), valid["label"]

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


def smv_classifier(train, valid):
    x_train, y_train = train.drop("label", 1), train["label"]
    x_data, y_data = valid.drop("label", 1), valid["label"]
    svc = svm.LinearSVC(multi_class="crammer_singer")
    t1 = time()
    parameters = {'C': [1, 10, 100, 1000]}
    grid = GridSearchCV(svc, parameters, cv=5)
    grid.fit(x_train, y_train)

    classifier = grid.best_estimator_
    predictions = classifier.predict(x_data)
    print("-----------------------------------------------")
    print(metrics.confusion_matrix(y_data, predictions))
    t2 = time()
    print("training time = {:.2f}".format(t2 - t1))
    print("-----------------------------------------------")


if __name__ == "__main__":
    path = "EMG_data/"
    subjects = os.listdir(path)
    subjects.sort()

    test_len = len(subjects)-8
    train_validation = subjects[:test_len]
    t_len = int(len(train_validation)*0.6)

    test_subjects = subjects[test_len:]
    train_subjects = train_validation[:t_len]
    validation_subjects = train_validation[t_len:]
    print(train_subjects)
    empty_data = pd.DataFrame(columns=["time", "channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "class"])
    train = load_data(train_subjects, path, empty_data)
    valid = load_data(validation_subjects, path, empty_data)
    # test = load_data(test_subjects, path, empty_data)
    train_windows = windows(train, 200, 200)
    valid_windows = windows(valid, 200, 200)
    print("train: {}".format(len(train)))
    print("train_windows: {}".format(len(train_windows)))
    cols = []
    for i in range(1, 9):
        cols.append("mean"+ str(i))
        cols.append("min" + str(i))
        cols.append("max" + str(i))
    cols.append("label")
    t_windows = pd.DataFrame(columns=cols)
    for window in train_windows:
        window_features = get_features(window)
        t_windows = t_windows.append(window_features, ignore_index=True)
    print("train_windows features: {}".format(len(t_windows)))
    v_windows = pd.DataFrame(columns=cols)
    for window in train_windows:
        window_features = get_features(window)
        v_windows = v_windows.append(window_features, ignore_index=True)
    # t_windows = t_windows[t_windows["label"] != 0]
    # v_windows = v_windows[v_windows["label"] != 0]
    print("-----------------------------------------------")
    print("---------------value counts--------------------")
    print(t_windows["label"].value_counts())
    print("-----------------------------------------------")
    print("train_windows features (0 removed): {}".format(len(t_windows)))
    perceptron_classifier(t_windows, v_windows)