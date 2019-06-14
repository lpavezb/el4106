from time import time
import numpy as np
import pandas as pd
import os


def load_data(subjects, path, data):
    ndata = data.copy()
    for subject in subjects:
        new_path = path + subject + "/"
        d1, d2 = os.listdir(new_path)
        ndata = ndata.append(pd.read_csv(new_path + d1, sep="\t"))
        ndata = ndata.append(pd.read_csv(new_path + d2, sep="\t"))
    return ndata


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
    train = load_data(train_subjects, "EMG_data/", empty_data)


