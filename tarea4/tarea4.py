from time import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import six

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def bench_k_means(estimator, name, data, n_clusters, labels):
    estimator.set_params(n_clusters=n_clusters)
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels, estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean')))


if __name__ == "__main__":
    df = pd.read_csv("Frogs_MFCCs.csv")
    species = df["Species"]
    features = df.drop("RecordID", 1).drop("Family", 1).drop("Genus", 1).drop("Species", 1)
    labels_unique = {}
    species_unique = species.unique()
    for i in range(len(species_unique)):
        labels_unique[species_unique[i]] = i

    labels = []
    for specie in species:
        labels.append(labels_unique[specie])

    labels_df = pd.DataFrame(data={"label": labels})
    kmean = KMeans(init='k-means++', n_init=10)
    bench_k_means(kmean, "k-means++", features, 10, labels)
