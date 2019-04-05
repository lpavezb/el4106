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
    train_set_by_class = {"g": train_g, "h": train_h}  # dictionary of train sets by class
    test_set_by_class = {"g": test_g, "h": test_h}  # dictionary of test sets by class
    return test_set, train_set, train_set_by_class, test_set_by_class


def plot(train_set, bins, title="train_set"):
    plt.suptitle(title, fontsize="x-large")
    for i in range(10):
        plt.subplot(5, 2, i+1)
        plt.hist(train_set[:, i], bins=bins, normed=True)
        plt.title("x{}".format(i))
    plt.subplots_adjust(hspace=1, wspace=0.35)
    plt.show()


def find_bin(x, bin_edges):
    res = 0
    for edge in bin_edges[1:len(bin_edges)-1]:
        if x < edge:
            return res
        res += 1
    return res


def histogram(data, bins):
    m = min(data)
    M = max(data)
    size = len(data)
    step = (M - m) / bins
    bin_edges = [m]
    for i in range(1, bins+1):
        bin_edges.append(bin_edges[i-1] + step)

    hist = np.zeros(bins)
    for value in data:
        bin = find_bin(value, bin_edges)
        hist[bin] += 1

    for i in range(len(hist)):
        hist[i] = hist[i] / size
    return hist, bin_edges


if __name__ == '__main__':
    g = 12332
    h = 6688
    bins = 30
    data = np.genfromtxt('magic04_label.data', delimiter=',')
    test, train, train_by_class, test_by_class = split_data(data)
    hist_g = []
    edges_g = []
    hist_h = []
    edges_h = []
    for i in range(10):
        hist, edges = histogram(train_by_class["g"][:, i], bins=bins)
        hist_g.append(hist)
        edges_g.append(edges)
        hist, edges = histogram(train_by_class["h"][:, i], bins=bins)
        hist_h.append(hist)
        edges_h.append(edges)

    p0 = 1
    p1 = 1
    for x in test_by_class["g"]:
        for caract in range(10):
            p0 = p0 * hist_g[caract][find_bin(x[caract], edges_g[caract])]
        for caract in range(10):
            p1 = p1 * hist_h[caract][find_bin(x[caract], edges_h[caract])]

        if p0/p1 >= 2:
            print("hadron")
        else:
            print("no hadron")
