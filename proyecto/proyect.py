from time import time
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import os


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

    empty_data = pd.DataFrame(columns=["time", "channel1", "channel2", "channel3", "channel4", "channel5", "channel6", "channel7", "channel8", "class"])
    train = load_data(train_subjects, path, empty_data)
    # test = load_data(test_subjects, path, empty_data)
    windows_list = windows(train, 200, 200)

    cols = []
    for i in range(1, 9):
        cols.append("mean"+ str(i))
        cols.append("min" + str(i))
        cols.append("max" + str(i))
    cols.append("label")
    windows_df = pd.DataFrame(columns=cols)
    for window in windows_list:
        window_features = get_features(window)
        windows_df = windows_df.append(window_features, ignore_index=True)

    with pd.option_context('display.max_rows', 20, 'display.max_columns', 10):
        print(windows_df)
