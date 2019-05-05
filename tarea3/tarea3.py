#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def load_file():
    file = open("Restaurant_Reviews_pl.tsv")
    ar = []
    for line in file:
        ar.append(line.split('\t'))
    for i in range(len(ar)):
        ar[i][1] = ar[i][1].replace('\n', '')
    n_ar = np.array(ar)
    return n_ar[1:, 0], n_ar[1:, 1]  # delete 'Review Liked' line


if __name__ == "__main__":
    text, labels= load_file()
    print(text)