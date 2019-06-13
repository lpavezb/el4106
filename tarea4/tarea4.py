from time import time
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


def bench_k_means(estimator, name, data, true_labels):
    n_clusters = estimator.get_params()["n_clusters"]
    t0 = time()
    estimator.fit(data)
    labels = estimator.labels_
    homo, compl, vmeas = metrics.homogeneity_completeness_v_measure(true_labels, labels)
    silhouette = metrics.silhouette_score(data, labels, metric='euclidean')
    print("{:20}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}".format(
        name,
        round(time() - t0, 3),
        round(homo, 3),
        round(compl, 3),
        round(vmeas, 3),
        round(silhouette, 3),
        n_clusters))


def bench_DBSCAN(epsilon, data, true_labels, extra_cluster):
    data_ = data.copy()
    metric_labels = true_labels[:]
    estimator = DBSCAN(eps=epsilon)
    t0 = time()
    estimator.fit(data_)
    labels = estimator.labels_
    clusters = set(labels)
    n_clusters_ = len(clusters)
    if not extra_cluster:
        est_labels = estimator.labels_
        indexs = []
        for i in range(len(est_labels)):
            if est_labels[i] == -1:
                indexs.append(i)

        for i in indexs[::-1]:
            labels = np.delete(labels, i, axis=0)
            metric_labels.pop(i)
            data_.drop(data_.index[i], inplace=True)
        if -1 in clusters:
            n_clusters_ -= 1
    else:
        last = n_clusters_
        for i in range(len(labels)):
            if labels[i] == -1:
                labels[i] = last

    if n_clusters_ <= 1:
        homo = "----"
        compl = "----"
        vmeas = "----"
        silhouette = "----"
    else:
        homo, compl, vmeas = metrics.homogeneity_completeness_v_measure(metric_labels, labels)
        homo = round(homo, 3)
        compl = round(compl, 3)
        vmeas = round(vmeas, 3)
        silhouette = round(metrics.silhouette_score(data_, labels, metric='euclidean'), 3)
    print("{:20}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}".format("DBSCAN_" + str(epsilon) + "_" + str(extra_cluster),
                                                                         round(time() - t0, 3),
                                                                         homo,
                                                                         compl,
                                                                         vmeas,
                                                                         silhouette,
                                                                         n_clusters_))


def bench_agglomerative_clustering(estimator, data, true_labels):
    t0 = time()
    estimator.fit(data)
    labels = estimator.labels_
    clusters = set(labels)
    n_clusters_ = len(clusters)

    if n_clusters_ <= 1:
        homo = "----"
        compl = "----"
        vmeas = "----"
        silhouette = "----"
    else:
        homo, compl, vmeas = metrics.homogeneity_completeness_v_measure(true_labels, labels)
        homo = round(homo, 3)
        compl = round(compl, 3)
        vmeas = round(vmeas, 3)
        silhouette = round(metrics.silhouette_score(data, labels, metric='euclidean'), 3)
    print("{:20}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}\t{:<10}".format("AGGC",
                                                                         round(time() - t0, 3),
                                                                         homo,
                                                                         compl,
                                                                         vmeas,
                                                                         silhouette,
                                                                         n_clusters_))


if __name__ == "__main__":
    np.random.seed(42)
    df = pd.read_csv("Frogs_MFCCs.csv")
    species = df["Species"]
    features = df.drop(["RecordID", "Family", "Genus", "Species"], 1)
    feature_names = features.columns.tolist()
    labels_unique = {}
    species_unique = species.unique()
    for i in range(len(species_unique)):
        labels_unique[species_unique[i]] = i

    labels = []
    for specie in species:
        labels.append(labels_unique[specie])

    n_samples, n_features = features.shape
    n_clusters = len(np.unique(labels))

    features = scale(features)
    features = pd.DataFrame(data=features, columns=feature_names)

    print(100 * '_')
    print("n_clusters: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))
    print(100 * '_')
    print("{:20}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}".format("init", "time", "homo", "compl", "v-meas", "silhouette",
                                                                   "n_clusters"))

    bench_k_means(KMeans(init='random', n_clusters=10), "KMeans_random", features, labels)
    bench_k_means(KMeans(init='k-means++', n_clusters=10), "KMeans_k-means++", features, labels)
    bench_DBSCAN(0.5, features, labels, False)
    bench_DBSCAN(0.7, features, labels, False)
    bench_DBSCAN(0.2, features, labels, False)
    bench_DBSCAN(0.5, features, labels, True)
    bench_DBSCAN(0.7, features, labels, True)
    bench_DBSCAN(0.2, features, labels, True)
    bench_agglomerative_clustering(AgglomerativeClustering(n_clusters=10), features, labels)

    print(100 * '_')

    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(features)

    pca_features = pd.DataFrame(data=pca_features, columns=['col1', 'col2'])
    print(48 * " " + "PCA")
    print(100*"_")
    n_samples, n_features = pca_features.shape
    print("n_clusters: %d, \t n_samples %d, \t n_features %d"
          % (n_clusters, n_samples, n_features))
    print(100 * '_')
    print("{:20}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}\t{:10}".format("init", "time", "homo", "compl", "v-meas",
                                                                   "silhouette",
                                                                   "n_clusters"))

    bench_k_means(KMeans(init='random', n_clusters=10), "random", pca_features, labels)
    bench_k_means(KMeans(init='k-means++', n_clusters=10), "k-means++", pca_features, labels)
    bench_DBSCAN(0.5, pca_features, labels, False)
    bench_DBSCAN(0.7, pca_features, labels, False)
    bench_DBSCAN(0.2, pca_features, labels, False)
    bench_DBSCAN(0.5, pca_features, labels, True)
    bench_DBSCAN(0.7, pca_features, labels, True)
    bench_DBSCAN(0.2, pca_features, labels, True)
    bench_agglomerative_clustering(AgglomerativeClustering(n_clusters=10), pca_features, labels)

    print(100 * '_')
