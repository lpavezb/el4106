#!/usr/bin/env python
import numpy as np
from matplotlib import pyplot as plt
import math


########################### DATA MANAGEMENT SECTION ###########################


def separate_by_class(data):
    g = np.array(range(11))  # this is needed for vstack function
    h = np.array(range(11))
    for x in data:
        if x[10] == 1:
            h = np.vstack((h, x))
        else:
            g = np.vstack((g, x))

    g = g[1:]  # delete the first row (range(11))
    h = h[1:]

    return g, h


def split_data(data):
    # g: 0
    # h: 1
    g, h = separate_by_class(data)

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
    return test_set, train_set


########################### HISTOGRAM MODEL SECTION ###########################

def plot_hists(hists, edges, bins, title="train_set"):
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


def get_histogram_rates(test, hist_g, edges_g, hist_h, edges_h, theta):
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


########################### GAUSSIAN MODEL SECTION ###########################

def gaussian(x, mu, cov):
    pi = math.pi
    k = len(x)
    diff = x - mu
    inv = np.linalg.inv(cov)
    det = np.linalg.det(cov)
    exp = -1 * 0.5 * ((diff.transpose()).dot(inv)).dot(diff)
    den = math.sqrt(math.pow(2 * pi, k) * det)
    return math.exp(exp) / den


def get_parameters(data):
    mu = np.array([])
    for i in range(10):
        mu = np.hstack((mu, np.mean(data[:, i])))
    cov = np.cov(data, rowvar=False)
    return mu, cov


def get_gaussian_rates(test, mu_g, cov_g, mu_h, cov_h, theta):
    TP = 0
    FN = 0
    FP = 0
    TN = 0

    for x in test:
        p0 = gaussian(x[:10], mu_g, cov_g)
        p1 = gaussian(x[:10], mu_h, cov_h)
        if p0 == 0: p0 = 0.00000000000001
        if p1 == 0: p1 = 0.00000000000001
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


########################### MAIN SECTION ###########################

def naive_bayes(bins, test, train, thetas):
    hist_g = []
    edges_g = []
    hist_h = []
    edges_h = []

    g, h = separate_by_class(train)
    for i in range(10):
        hist, edges = histogram(g[:, i], bins)
        hist_g.append(hist)
        edges_g.append(edges)
        hist, edges = histogram(h[:, i], bins)
        hist_h.append(hist)
        edges_h.append(edges)

    roc = []
    for theta in thetas:
        roc.append(get_histogram_rates(test, hist_g, edges_g, hist_h, edges_h, theta))
    xs = [x[0] for x in roc]
    ys = [x[1] for x in roc]
    plt.plot(xs, ys, label="histogram model, {} bins".format(bins))


def gauss_bayes(test, train, thetas):
    g, h = separate_by_class(train)

    mu_h, cov_h = get_parameters(h[:, :10])  # delete last column
    mu_g, cov_g = get_parameters(g[:, :10])
    roc = []
    for theta in thetas:
        roc.append(get_gaussian_rates(test, mu_g, cov_g, mu_h, cov_h, theta))
    xs = [x[0] for x in roc]
    ys = [x[1] for x in roc]
    plt.plot(xs, ys, label="gauss model")


def tarea1(entrenamiento, prueba):
    bins = 60
    thetas = np.hstack((np.arange(0, 1, 0.01), np.arange(1, 100, 1)))
    naive_bayes(bins, prueba, entrenamiento, thetas)
    gauss_bayes(prueba, entrenamiento, thetas)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve, number of thetas = {}".format(len(thetas)))
    plt.legend(loc="lower right")
    plt.show()


if __name__ == '__main__':
    data = np.genfromtxt('magic04_label.data', delimiter=',')
    test, train = split_data(data)
    tarea1(train, test)
