#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt


def split_data(dat):
    # g: 0
    # h: 1
    g = np.array(range(11))  # this is needed for vstack function
    h = np.array(range(11))
    for x in data:
        if x[10] == 1:
            h = np.vstack((h, x))
        else:
            g = np.vstack((g, x))

    g = g[1:]  # delete the first row (range(11))
    h = h[1:]

    # shuffle data
    np.random.shuffle(g)
    np.random.shuffle(h)

    test_g_len = len(g) * 20 // 100
    test_h_len = len(h) * 20 // 100

    # separate between test set and train set for each class (20% - 80%)
    test_g = np.array(g[:test_g_len])
    train_g = np.array(g[test_g_len:])
    test_h = np.array(h[:test_h_len])
    train_h = np.array(h[test_h_len:])

    train_set = np.vstack((train_g, train_h))  # combine the sets of each train set class into one train set
    test_set = np.vstack((test_g, test_h))  # combine the sets of each test set class into one test set
    train_set_by_class = {"g": train_g, "h": train_h}  # dictionary of train sets by class
    test_set_by_class = {"g": test_g, "h": test_h}  # dictionary of test sets by class
    return test_set, train_set, train_set_by_class, test_set_by_class


def plot_np_hist(train_set, bins, title="train_set"):
    plt.suptitle(title, fontsize="x-large")
    for i in range(10):
        plt.subplot(5, 2, i + 1)
        plt.hist(train_set[:, i], bins=bins, density=True)
        plt.title("x{}".format(i))
    plt.subplots_adjust(hspace=1, wspace=0.35)


def plot_manual_hist(hists, edges, bins, title="train_set"):
    plt.suptitle(title, fontsize="x-large")
    for i in range(10):
        bar_width = edges[i][1] - edges[i][0]
        plt.subplot(5, 2, i + 1)
        plt.bar(edges[i][:bins], hists[i], width=bar_width)
        plt.title("x{}".format(i))
    plt.subplots_adjust(hspace=1, wspace=0.35)


def find_bin(x, bin_edges):
    res = 0
    for edge in bin_edges[1:len(bin_edges) - 1]:
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
    for i in range(1, bins + 1):
        bin_edges.append(bin_edges[i - 1] + step)

    hist = np.zeros(bins)
    for value in data:
        b = find_bin(value, bin_edges)
        hist[b] += 1

    for i in range(len(hist)):
        hist[i] = hist[i] / size
    return hist, bin_edges


def get_prob(hist, edges, x):
    s = len(edges)
    zero = 0.00000000000001
    res = zero
    if edges[0] <= x <= edges[s - 1]:
        res = hist[find_bin(x, edges)]
        if res == 0:
            res = zero
    return res


def get_rates(test, hist_g, edges_g, hist_h, edges_h, theta):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for x in test:
        p0 = p1 = 1
        for caract in range(10):
            p0 = p0 * get_prob(hist_g[caract], edges_g[caract], x[caract])
        for caract in range(10):
            p1 = p1 * get_prob(hist_h[caract], edges_h[caract], x[caract])

        if p1 / p0 >= theta:  # classifier: hadron
            if x[10] == 1:  # real: hadron
                TP += 1
            else:  # real: gamma
                FP += 1
        else:  # classifier: gamma
            if x[10] == 1:  # real: hadron
                FN += 1
            else:  # real: gamma
                TN += 1

    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [FPR, TPR]


def naive_bayes(bins, test, train_by_class):
    hist_g = []
    edges_g = []
    hist_h = []
    edges_h = []
    for i in range(10):
        hist, edges = histogram(train_by_class["g"][:, i], bins)
        hist_g.append(hist)
        edges_g.append(edges)
        hist, edges = histogram(train_by_class["h"][:, i], bins)
        hist_h.append(hist)
        edges_h.append(edges)

    values = np.hstack((np.arange(0, 1, 0.05), np.array(range(1, 100))))

    roc = []
    for theta in values:
        roc.append(get_rates(test, hist_g, edges_g, hist_h, edges_h, theta))
    xs = [x[0] for x in roc]
    ys = [x[1] for x in roc]
    plt.plot(xs, ys, label="{} bins".format(bins))
    plt.legend(loc="lower right")


def gaussian(x, mu, cov):
    pass


if __name__ == '__main__':
    data = np.genfromtxt('magic04_label.data', delimiter=',')
    test, train, train_by_class, test_by_class = split_data(data)

    # bins = range(40, 101, 10)
    # for b in bins:
    #     naive_bayes(b, test, train_by_class)
    # plt.show()

