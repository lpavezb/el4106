import os
import sys
import math
import numpy as np
import pandas as pd
from time import time
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from scipy.stats import kurtosis, skew
from classifiers import svm_classifier, mlp_classifier


def root_mean_square(data):
    square = 0
    for i in range(len(data)):
        square += (data[i] ** 2)

    root = math.sqrt(square / len(data))
    return root


def plot_data(subject):
    path = "EMG_data/"
    columns = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "class"]
    empty_data = pd.DataFrame(columns=columns)
    x_s, y_s = load_data([subject], path, empty_data)
    plt.figure()
    plt.suptitle("subject {}".format(subject), fontsize="x-large")
    plt.subplot(5, 1, 1)
    plt.plot(range(len(y_s)), y_s)
    plt.title("class")
    for i in range(8):
        plt.subplot(5, 2, i + 3)
        c = "channel{}".format(i + 1)
        plt.plot(range(len(x_s)), x_s[c])
        plt.title(c)
    plt.subplots_adjust(hspace=1, wspace=0.35)


def windows(data, width, step):
    start = 0
    windows_list = []
    while (start + width) < len(data):
        data_slice = data[start:start + width].reset_index()
        if data_slice["class"][0] == data_slice["class"][width - 1]:
            windows_list.append(data_slice)
        start += step
    return windows_list


def load_data(subjects, path, data, drop7=True):
    ndata = data.copy()
    for subject in subjects:
        new_path = path + subject + "/"
        for file in os.listdir(new_path):
            ndata = ndata.append(pd.read_csv(new_path + file, sep="\t"), sort=True)
    if drop7:
        ndata = ndata[ndata["class"] != 7]
    ndata = ndata.drop("time", 1)
    return ndata.drop("class", 1), ndata["class"]


def get_features(window):
    features = {}
    for i in range(1, 9):
        max_value = np.min(window["channel" + str(i)])
        min_value = np.max(window["channel" + str(i)])
        features["mean" + str(i)] = np.mean(window["channel" + str(i)])
        features["variance" + str(i)] = np.var(window["channel" + str(i)])
        features["min" + str(i)] = min_value
        features["max" + str(i)] = max_value
        features["range" + str(i)] = max_value - min_value
        features["RMS" + str(i)] = root_mean_square(window["channel" + str(i)])
        features["skew" + str(i)] = skew(window["channel" + str(i)])
        features["kurtosis" + str(i)] = kurtosis(window["channel" + str(i)])
    features["label"] = window["class"][0]
    return features


def generate_csvs():
    load_and_save_csv(width=100, step=100)
    load_and_save_csv(width=100, step=200)
    load_and_save_csv(width=100, step=400)

    load_and_save_csv(width=200, step=100)
    load_and_save_csv(width=200, step=200)
    load_and_save_csv(width=200, step=400)

    load_and_save_csv(width=400, step=100)
    load_and_save_csv(width=400, step=200)
    load_and_save_csv(width=400, step=400)


def load_and_save_csv(width=200, step=200):
    path = "EMG_data/"
    subjects = os.listdir(path)
    subjects.sort()

    test_len = len(subjects) - 8

    test_subjects = subjects[test_len:]
    train_validation_subjects = subjects[:test_len]

    print("train and validation subjects: {}".format(train_validation_subjects))
    print("test subjects: {}".format(test_subjects))

    columns = ["channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "class"]
    empty_data = pd.DataFrame(columns=columns)

    print(100 * '-')
    print(44 * '-' + "loading data" + 44 * '-')
    t = time()
    x_train_valid, y_train_valid = load_data(train_validation_subjects, path, empty_data)
    x_test, y_test = load_data(test_subjects, path, empty_data)

    print("train_validation samples: {}".format(len(x_train_valid)))
    print("test_samples: {}".format(len(x_test)))
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(40 * '-' + "calculating windows" + 41 * '-')
    t = time()
    train_valid_values = np.hstack((x_train_valid, y_train_valid.to_numpy().reshape(len(x_train_valid), 1)))
    test_values = np.hstack((x_test, y_test.to_numpy().reshape(len(x_test), 1)))
    train_valid = pd.DataFrame(columns=columns, data=train_valid_values)
    test = pd.DataFrame(columns=columns, data=test_values)

    train_valid_windows_list = windows(train_valid, width, step)
    test_windows_list = windows(test, width, step)

    if width == 100:  # error al calcular caracteristicas a ventanas de largo 100 al incluir label 0
        train_valid_windows_list = [x for x in train_valid_windows_list if x["class"][0] != 0]
        test_windows_list = [x for x in test_windows_list if x["class"][0] != 0]

    np.random.shuffle(train_valid_windows_list)
    train_len = int(len(train_valid_windows_list) * 0.8)
    train_windows = train_valid_windows_list[:train_len]
    valid_windows = train_valid_windows_list[train_len:]
    test_windows = test_windows_list

    print("train_windows: {}".format(len(train_windows)))
    print("validation_windows: {}".format(len(valid_windows)))
    print("test_windows: {}".format(len(test_windows)))
    print("total_windows: {}".format(len(train_windows) + len(valid_windows) + len(test_windows)))
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(40 * '-' + "calculating features" + 40 * '-')

    features = ["mean", "variance", "min", "max", "range", "RMS", "skew", "kurtosis"]
    print("features: {}".format(features))
    cols = []
    for i in range(1, 9):
        for feature in features:
            cols.append(feature + str(i))
    cols.append("label")

    print("calculating train features")
    train_features = pd.DataFrame(columns=cols)
    for window in train_windows:
        window_features = get_features(window)
        train_features = train_features.append(window_features, ignore_index=True)

    print("calculating validation features")
    validation_features = pd.DataFrame(columns=cols)
    for window in valid_windows:
        window_features = get_features(window)
        validation_features = validation_features.append(window_features, ignore_index=True)

    print("calculating test features")
    test_features = pd.DataFrame(columns=cols)
    for window in test_windows:
        window_features = get_features(window)
        test_features = test_features.append(window_features, ignore_index=True)

    train_name = "train_features_" + str(width) + "_" + str(step) + ".csv"
    validation_name = "validation_features_" + str(width) + "_" + str(step) + ".csv"
    test_name = "test_features_" + str(width) + "_" + str(step) + ".csv"

    train_features.to_csv(train_name, index=None, header=True)
    validation_features.to_csv(validation_name, index=None, header=True)
    test_features.to_csv(test_name, index=None, header=True)


def gesture_classifier_test():
    sizes_list = ["100", "200", "400"]
    print(100 * "-")
    print(100 * "-")
    print(100 * "-")
    print("gesture classifier")
    svm_results = {}
    mlp_results = {}
    ks = [20, 30, 40, 50, 64]
    for k in ks:
        svm_results[k] = {}
        mlp_results[k] = {}
    size_w = len(sizes_list)
    size_s = len(sizes_list)
    for w in sizes_list:
        for s in sizes_list:
            print(100 * '-')
            print("window width: {}".format(w))
            print("window step: {}".format(s))
            train_file = "train_features_" + w + "_" + s + ".csv"
            train_features = pd.read_csv(train_file, sep=",")
            train_features = train_features.loc[train_features['label'] != 0].to_numpy()

            validation_file = "validation_features_" + w + "_" + s + ".csv"
            validation_features = pd.read_csv(validation_file, sep=",")
            validation_features = validation_features.loc[validation_features['label'] != 0].to_numpy()

            print(100 * '-')
            print(36 * '-' + "Standardization of datasets" + 37 * '-')
            t = time()
            train_nc = train_features.shape[1] - 1
            x_train = train_features[:, :train_nc]
            y_train = train_features[:, train_nc]

            valid_nc = validation_features.shape[1] - 1
            x_valid = validation_features[:, :valid_nc]
            y_valid = validation_features[:, valid_nc]

            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_valid = scaler.transform(x_valid)
            print("time: {:.2f}".format(time() - t))
            print(100 * '-')

            print(41 * '-' + "feature selection" + 42 * '-')

            for k in ks:
                print("{} features".format(k))
                selector = SelectKBest(mutual_info_classif, k=k)
                selector.fit(x_train, y_train)
                train = selector.transform(x_train)
                valid = selector.transform(x_valid)
                print(100 * "-")
                print("MLP classifier")
                mlp_results[k][w + 'x' + s] = int(round(mlp_classifier(train, y_train, valid, y_valid)[0] * 100))
                print(100 * "-")
                print("SVM classifier")
                svm_results[k][w + 'x' + s] = int(round(svm_classifier(train, y_train, valid, y_valid)[0] * 100))
                print(100 * "-")
    return [svm_results, mlp_results], [size_w, size_s]


def pause_detector_test():
    print(100 * "-")
    print(100 * "-")
    print(100 * "-")
    print("pause vs gesture classifier")
    sizes_list = ["100", "200", "400"]
    svm_results = {}
    mlp_results = {}
    ks = [20, 30, 40, 50, 64]
    for k in ks:
        svm_results[k] = {}
        mlp_results[k] = {}
    size_w = len(sizes_list[1:])
    size_s = len(sizes_list)
    for w in sizes_list[1:]:
        for s in sizes_list:
            print(100 * '-')
            print("window width: {}".format(w))
            print("window step: {}".format(s))
            train_file = "train_features_" + w + "_" + s + ".csv"
            train_features = pd.read_csv(train_file, sep=",")
            train_features.loc[train_features['label'] != 0, 'label'] = 1
            train_features = train_features.to_numpy()

            validation_file = "validation_features_" + w + "_" + s + ".csv"
            validation_features = pd.read_csv(validation_file, sep=",")
            validation_features.loc[validation_features['label'] != 0, 'label'] = 1
            validation_features = validation_features.to_numpy()

            print(36 * '-' + "Standardization of datasets" + 37 * '-')
            t = time()
            train_nc = train_features.shape[1] - 1
            x_train = train_features[:, :train_nc]
            y_train = train_features[:, train_nc]

            valid_nc = validation_features.shape[1] - 1
            x_valid = validation_features[:, :valid_nc]
            y_valid = validation_features[:, valid_nc]

            scaler = StandardScaler()
            scaler.fit(x_train)
            x_train = scaler.transform(x_train)
            x_valid = scaler.transform(x_valid)
            print("time: {:.2f}".format(time() - t))
            print(100 * '-')

            print(41 * '-' + "feature selection" + 42 * '-')
            for k in ks:

                print("{} features".format(k))
                selector = SelectKBest(mutual_info_classif, k=k)
                selector.fit(x_train, y_train)
                train = selector.transform(x_train)
                valid = selector.transform(x_valid)
                print(100 * "-")
                print("MLP classifier")
                mlp_results[k][w + 'x' + s] = int(round(mlp_classifier(train, y_train, valid, y_valid)[0] * 100))
                print(100 * "-")
                print("SVM classifier")
                svm_results[k][w + 'x' + s] = int(round(svm_classifier(train, y_train, valid, y_valid)[0] * 100))
                print(100 * "-")
    return [svm_results, mlp_results], [size_w, size_s]


# https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
def plot_confussion_matrix(confm, title="confussion_matrix", classes=range(1, 7)):
    # Only use the labels that appear in the data
    fig, ax = plt.subplots()
    im = ax.imshow(confm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(confm.shape[1]),
           yticks=np.arange(confm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = confm.max() / 2.
    for i in range(confm.shape[0]):
        for j in range(confm.shape[1]):
            ax.text(j, i, format(confm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if confm[i, j] > thresh else "black")
    fig.tight_layout()


def print_results(accuracy_dict, size):
    rows, cols = size
    ks = list(accuracy_dict.keys())
    ks.sort()
    windows = list(accuracy_dict[ks[0]].keys())
    windows.sort()
    for k in ks:
        rows_list = []
        s = 0
        for i in range(rows):
            rows_list.append(windows[s:s+cols])
            s += cols
        print(100 * '-')
        print("k = {}".format(k))
        print("--- | 100 200 400")
        for r in range(rows):
            r_name = (rows_list[r][0].split("x"))[0]
            print(r_name + " | " + "  ".join(str(accuracy_dict[k][x]) for x in rows_list[r]))

    print(100 * '-')


def both_classifiers_test():
    np.set_printoptions(threshold=sys.maxsize)
    np.random.seed(42)
    w = '400'
    s = '100'
    k = 40
    print(100 * '-')
    print("window width: {}".format(w))
    print("window step: {}".format(s))

    train_file = "train_features_" + w + "_" + s + ".csv"
    train_features = pd.read_csv(train_file, sep=",")
    train_gesture_y = train_features['label'].copy().to_numpy()
    train_features.loc[train_features['label'] != 0, 'label'] = 1
    train_features = train_features.to_numpy()

    test_file = "test_features_" + w + "_" + s + ".csv"
    test_features = pd.read_csv(test_file, sep=",")
    test_gesture_y = test_features['label'].copy().to_numpy()
    test_features.loc[test_features['label'] != 0, 'label'] = 1
    test_features = test_features.to_numpy()

    print(36 * '-' + "Standardization of datasets" + 37 * '-')
    t = time()
    train_nc = train_features.shape[1] - 1
    x_train = train_features[:, :train_nc]
    y_train = train_features[:, train_nc]

    test_nc = test_features.shape[1] - 1
    x_test = test_features[:, :test_nc]
    y_test = test_features[:, test_nc]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')
    print(41 * '-' + "feature selection" + 42 * '-')
    print("{} features".format(k))
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(x_train, y_train)
    train = selector.transform(x_train)
    test = selector.transform(x_test)
    print(100 * "-")
    print("MLP classifier")
    acc, pause_classifier = mlp_classifier(train, y_train, test, y_test)
    pause_predictions = pause_classifier.predict(test)
    real_y_pred = []
    x_pred = []
    for i in range(len(pause_predictions)):
        if pause_predictions[i] == 1:
            real_y_pred.append(test_gesture_y[i])
            x_pred.append(test[i])

    acc, gesture_classifier = mlp_classifier(train, train_gesture_y, x_pred, real_y_pred)
    gesture_predictions = gesture_classifier.predict(x_pred)
    print(np.unique(gesture_predictions))
    confm = metrics.confusion_matrix(real_y_pred, gesture_predictions)
    confm_diagonal = np.diag(confm)
    accuracy = confm_diagonal.sum() / confm.sum()
    print(100 * "-")
    print(confm)
    print("Accuracy: {:.4f}".format(accuracy))
    print(100 * "-")
    plot_confussion_matrix(confm, title="Pause detector and Gesture Classifier", classes=range(7))
    plt.savefig("pause_detector_and_gesture_classifier_test")


def run_tests_and_print_results():
    results, size = gesture_classifier_test()
    print(100 * '-')
    print("smv results")
    print_results(results[0], size)
    print(100 * '-')
    print("mlp results")
    print_results(results[1], size)
    print(100 * '-')
    print(100 * '-')
    results, size = pause_detector_test()
    print(100 * '-')
    print("smv results")
    print_results(results[0], size)
    print(100 * '-')
    print("mlp results")
    print_results(results[1], size)
    print(100 * '-')


def classifiers_test():
    print("gesture classifier")
    w = '400'
    s = '100'
    k = 40

    print(100 * '-')
    print("window width: {}".format(w))
    print("window step: {}".format(s))
    train_file = "train_features_" + w + "_" + s + ".csv"
    train_features = pd.read_csv(train_file, sep=",")
    train_features = train_features.loc[train_features['label'] != 0].to_numpy()

    test_file = "test_features_" + w + "_" + s + ".csv"
    test_features = pd.read_csv(test_file, sep=",")
    test_features = test_features.loc[test_features['label'] != 0].to_numpy()

    print(100 * '-')
    print(36 * '-' + "Standardization of datasets" + 37 * '-')
    t = time()
    train_nc = train_features.shape[1] - 1
    x_train = train_features[:, :train_nc]
    y_train = train_features[:, train_nc]

    test_nc = test_features.shape[1] - 1
    x_test = test_features[:, :test_nc]
    y_test = test_features[:, test_nc]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(41 * '-' + "feature selection" + 42 * '-')

    print("{} features".format(k))
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)
    print(100 * "-")
    print("MLP classifier")
    acc, gesture_classifier = mlp_classifier(x_train, y_train, x_test, y_test)
    acc = int(round(acc * 100))
    print("accuracy: {}".format(acc))
    confm = metrics.confusion_matrix(y_test, gesture_classifier.predict(x_test))
    plot_confussion_matrix(confm, title="Gesture Classifier", classes=range(1, 7))
    plt.savefig("gesture_classifier_test")
    print(100 * "-")

    print(100 * "-")
    print(100 * "-")
    print(100 * "-")
    print("pause vs gesture classifier")

    k = 64

    print(100 * '-')
    print("window width: {}".format(w))
    print("window step: {}".format(s))
    train_file = "train_features_" + w + "_" + s + ".csv"
    train_features = pd.read_csv(train_file, sep=",")
    train_features.loc[train_features['label'] != 0, 'label'] = 1
    train_features = train_features.to_numpy()

    test_file = "test_features_" + w + "_" + s + ".csv"
    test_features = pd.read_csv(test_file, sep=",")
    test_features.loc[test_features['label'] != 0, 'label'] = 1
    test_features = test_features.to_numpy()

    print(36 * '-' + "Standardization of datasets" + 37 * '-')
    t = time()
    train_nc = train_features.shape[1] - 1
    x_train = train_features[:, :train_nc]
    y_train = train_features[:, train_nc]

    test_nc = test_features.shape[1] - 1
    x_test = test_features[:, :test_nc]
    y_test = test_features[:, test_nc]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    print("time: {:.2f}".format(time() - t))
    print(100 * '-')

    print(41 * '-' + "feature selection" + 42 * '-')

    print("{} features".format(k))
    selector = SelectKBest(mutual_info_classif, k=k)
    selector.fit(x_train, y_train)
    x_train = selector.transform(x_train)
    x_test = selector.transform(x_test)
    print(100 * "-")
    print("MLP classifier")
    acc, pause_detector = mlp_classifier(x_train, y_train, x_test, y_test)
    acc = int(round(acc * 100))
    print("accuracy: {}".format(acc))
    confm = metrics.confusion_matrix(y_test, pause_detector.predict(x_test))
    plot_confussion_matrix(confm, title="Pause Detector", classes=range(2))
    plt.savefig("pause_detector_test")
    print(100 * "-")


def main():
    print(100 * '-')
    # generate_csvs()  # PUEDE TOMAR MUCHO TIEMPO
    # run_tests_and_print_results()
    classifiers_test()
    # both_classifiers_test()
    print(100 * '-')


if __name__ == "__main__":
    main()
