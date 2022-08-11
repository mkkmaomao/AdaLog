# from pyclustering.cluster.bsas import bsas_visualizer
# from pyclustering.cluster.mbsas import mbsas
# from pyclustering.utils import read_sample
# from pyclustering.samples.definitions import SIMPLE_SAMPLES
import os
import pickle
import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Prepare algorithm's parameters.


from sklearn.cluster import Birch


def load_data(n_components):
    # mode = "rb"
    with open(os.path.join(emb_dir,"iforest-train-ws20-per8.pkl"), mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)

    x_sum_tr = []
    for i in range(len(x_tr)):
        x_sum_tr.append(np.sum(x_tr[i], axis=0))
    x_sum_tr = np.array(x_sum_tr)
    print(x_sum_tr.shape)
    print(type(x_sum_tr))
    pca = PCA(n_components=n_components)
    x = pca.fit_transform(x_sum_tr)

    return x, y_tr


def MBSAS():
    # Create instance of MBSAS algorithm.
    max_clusters = 50
    threshold = 1.0
    sample = load_data()
    mbsas_instance = mbsas(sample, max_clusters, threshold)
    mbsas_instance.process()

    # Get clusters results.
    clusters = mbsas_instance.get_clusters()
    representatives = mbsas_instance.get_representatives()

    print("clusters: ", len(clusters))  # 50
    print("representatives: ", len(representatives))  # 50
    print("cluster[0]:", clusters[0])
    print(len(clusters[0]))  # 77243
    print("representatives[0]:", representatives[0])
    print(len(representatives[0]))  # 50


def Birch():
    X = load_data()
    brc = Birch(n_clusters=None)
    pre = brc.fit(X)
    pre = brc.predict(X)
    print(pre)

def predict_tag(inputs, y, labels):
        '''
        :param inputs: all input
        :param y: label.
        :param labels: cluster label.
        :return: predicted label for each line of inputs, labeled normal ones included.
        :return: normal id
        '''
        normal_cores = set()
        predicted = []
        normal_ids=[]
        
        assert len(inputs) == len(labels)
        inputs = np.asarray(inputs, dtype=float)
        normal_matrix = []
        labeledInst = []

        for idx in range(len(inputs)):
            if y[idx]==0:
                labeledInst.append(inputs[idx])
                if len(labeledInst) < len(inputs) / 2:
                    normal_matrix.append(inputs[idx,:])
                    normal_ids.append(idx)
                    if labels[idx] != -1:
                        normal_cores.add(labels[idx])

        print('Normal clusters are: ' + str(normal_cores))
        normal_matrix = np.asarray(normal_matrix, dtype=float)
        print('Shape of normal matrix: %d x %d' % (normal_matrix.shape[0], normal_matrix.shape[1]))

        by_normal_core_normal = 0
        by_normal_core_anomalous = 0
        by_dist_normal = 0
        by_dist_anomalous = 0

        for id, predict_cluster in enumerate(labels):
            if id in normal_ids:
                # Add labeled normals as predicted normals to formalize the output format for other modules.
                predicted.append('Normal')
                continue
            if predict_cluster in normal_cores:
                by_normal_core_normal += 1
                predicted.append('Normal')
            elif predict_cluster == -1:
                cur_repr = inputs[id]
                dists = cdist([cur_repr], normal_matrix)
                if dists.min() == 0:
                    by_dist_normal += 1
                    predicted.append('Normal')
                else:
                    by_dist_anomalous += 1
                    predicted.append('Anomalous')

                pass
            else:
                by_normal_core_anomalous += 1
                predicted.append('Anomalous')
        print('Found %d normal, %d anomalous by normal clusters' % (by_normal_core_normal, by_normal_core_anomalous))
        print('Found %d normal, %d anomalous by minimum distances' % (by_dist_normal, by_dist_anomalous))
        return predicted, normal_ids

def Hdbscan(x, y):
    import hdbscan
    from sklearn.decomposition import FastICA
    import random

    transformer = FastICA(n_components=50, random_state=0)
    X_transformed = transformer.fit_transform(x)
    clusterer = hdbscan.HDBSCAN(algorithm='best', alpha=1.0, approx_min_span_tree=True,
                                gen_min_span_tree=False, metric='euclidean', min_cluster_size=100, min_samples=100,
                                p=None, core_dist_n_jobs=10)
    clusterer.fit(X_transformed)
    labels = clusterer.labels_
    outliers = clusterer.outlier_scores_.tolist()
    # prob = clusterer.probabilities_.tolist()

    (predicts, normal_ids)=predict_tag(X_transformed, y, labels)
    assert len(X_transformed) == len(labels)

    labeled_inst = []
    TP, TN, FN, FP = 0, 0, 0, 0

    AAA, BBB = [], []
    for idx, inst, label, predict, outlier in zip(range(len(X_transformed)), X_transformed, labels, predicts, outliers):
        if idx in normal_ids:
            type = 'labeled'
            labeled_inst.append((idx, label, 'Normal', outlier, y[idx], type, 0))
            idx += 1
            continue

        type = 'unlabeled'
        if label == -1:
            # -1 cluster, all instances should have confidence 0
            confidence = 0
            pass
        else:
            # other clusters, instances should have confidence according to the outlier score.
            confidence = 1 if np.isnan(outlier) else outlier

        if predict == 'Normal':
            if y[idx] == 0:
                TN += 1
            else:
                FN += 1
            pass
        else:
            if y[idx] == 1:
                TP += 1
            else:
                FP += 1
            pass

        labeled_inst.append((idx, label, predict, outlier, y[idx], type, confidence))
        # index, label, predicts, outlier, ifAnomaly, ifLabeled, confidence

# ==============================for finding an example ==============================
        if (label == -1) & (predict == 'Normal'):
            if (y[idx] == 1):
                AAA.append(idx)


        elif (label == -1) & (predict == 'Anomalous'):
            if (y[idx] == 0):
                BBB.append(idx)

        idx += 1


    print("finished!")
    print("label=-1 and predict the abnomaly as a normal one: ", AAA)
    print("label=-1 and predict the normal as abnomaly: ", BBB)
    print("BBB length: ", len(BBB))
    print("FN: ", FN)
    # with open("./PLELog_examples.pkl", mode="wb") as f:
    #     pickle.dump(BBB, f)
    return TP, TN, FN, FP


def metricforHDBSCAN(TP, TN, FN, FP ):

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    Spec = TN / (TN+FP)
    if TP == 0:
        F1 = 0
    else:
        F1 = 2 * Precision * Recall / (Precision + Recall)

    print("tp: ", TP)
    print("fn: ", FN)
    print("fp: ", FP)
    print("tn: ", TN)
    print("Spec: ", Spec)
    print("Recall: ", Recall)
    print("Precision: ", Precision)
    print("F1: ", F1)

if __name__ == '__main__':
    emb_dir = '../data/embedding/BGL'
    n_components = 50
    save_path = '../clusters/BGL/PLELog'
    dataset = 'BGL'
    x, y = load_data(100)  # fastICA -> 50
    TP, TN, FN, FP = Hdbscan(x, y)
    metricforHDBSCAN(TP, TN, FN, FP)
