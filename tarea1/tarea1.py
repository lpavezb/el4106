#!/usr/bin/env python
import numpy as np

def splitData(dat):
    # g: 0
    # h: 1
    g_len = 0
    #find g class quantity (data is sorted by class)
    for i in range(len(dat)):
        if dat[i][10] == 1:
            g_len = i
            break

    g = np.array(dat[:g_len, :10]) # g class data
    h = np.array(dat[g_len:, :10]) # h class data

    # shufle data
    np.random.shuffle(g)
    np.random.shuffle(h)

    test_g_len = len(g)*20/100
    test_h_len = len(h)*20/100

    # separate between test set and train set for each class (20% - 80%)
    test_g  = np.array(g[:test_g_len])
    train_g = np.array(g[test_g_len:])
    test_h  = np.array(h[:test_h_len])
    train_h = np.array(h[test_h_len:, :])

    train_set = np.array(np.vstack((train_g, train_h))) # combine the sets of each train set class into one train set
    test_set = np.array(np.vstack((test_g, test_h))) # combine the sets of each test set class into one test set
    train_by_class = {0: train_g, 1: train_h} # dictionary of train sets by class
    return train_set, train_set, train_by_class


if __name__ == '__main__':
    dat = np.genfromtxt('magic04_label.data', delimiter=',')
    test_set, train_set , train_by_class = splitData(dat)
    g_hist = np.histogramdd(train_by_class[0], bins=2)