# -*- coding: utf-8 -*-

import numpy as np


def load_file():
    file = open("Restaurant_Reviews_pl.tsv")
    ar = []
    for line in file:
        ar.append(line.split('\t'))
    return ar[1:]  # delete 'Review Liked' line


if __name__ == "__main__":
    print(np.array(load_file()))
