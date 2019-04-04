#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt


def split_data(dat):
    # g: 0
    # h: 1
    g_len = 12332
    g = np.array(dat[:g_len, :10])  # g class data
    h = np.array(dat[g_len:, :10])  # h class data

    # shufle data
    np.random.shuffle(g)
    np.random.shuffle(h)

    test_g_len = len(g) * 20 // 100
    test_h_len = len(h) * 20 // 100

    # separate between test set and train set for each class (20% - 80%)
    test_g = np.array(g[:test_g_len])
    train_g = np.array(g[test_g_len:])
    test_h = np.array(h[:test_h_len])
    train_h = np.array(h[test_h_len:])

    train_set = np.array(np.vstack((train_g, train_h)))  # combine the sets of each train set class into one train set
    test_set = np.array(np.vstack((test_g, test_h)))  # combine the sets of each test set class into one test set
    train_set_by_class = {0: train_g, 1: train_h}  # dictionary of train sets by class
    test_set_by_class = {0: test_g, 1: test_h}  # dictionary of test sets by class
    return test_set, train_set, train_set_by_class, test_set_by_class


def plot(train_set, bins, title="train_set"):
    plt.suptitle(title, fontsize="x-large")
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.hist(train_set[:, i], bins=bins, density=True)
        plt.title("x{}".format(i))
    plt.subplots_adjust(hspace=1, wspace=0.35)
    plt.show()


if __name__ == '__main__':
    g = 12332
    h = 6688
    bins = 40
    data = np.genfromtxt('magic04_label.data', delimiter=',')
    test, train, train_by_class, test_by_class = split_data(data)
    hist_g = []
    hist_h = []
    for i in range(10):
        hist_g.append(np.histogram(train_by_class[0][:, i], bins=bins, density=True))
        hist_h.append(np.histogram(train_by_class[1][:, i], bins=bins, density=True))

    p0 = g / (g + h)
    p1 = h / (g + h)
