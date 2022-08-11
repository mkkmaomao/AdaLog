import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import sys
import random
from sklearn.decomposition import PCA
from collections import Counter
import math


def kmeans():
    with open("../data/embedding/BGL/iforest-train-200-per{}.pkl".format(percent), mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)

    x_sum_tr = []
    for i in range(len(x_tr)):
        x_sum_tr.append(np.sum(x_tr[i], axis=0))
    x_sum_tr = np.array(x_sum_tr)
    print(x_sum_tr.shape)
    print(type(x_sum_tr))

    pca = PCA(n_components=n_components)
    x = pca.fit_transform(x_sum_tr)
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=50, max_iter=100, tol=0.0001, verbose=0, random_state=None,
                    copy_x=True, algorithm='auto').fit(x)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    dist = kmeans.transform(x)
    print("length of labels: ", len(labels))
    print("centers shape: ", (np.array(centers)).shape)
    print("dist shape: ", (np.array(dist)).shape)


    with open('./clusters/{}/{}/per={}/classIndex_k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), 'wb') as f:
        pickle.dump(labels, f) # should be [1, 460048]
    with open('./clusters/{}/{}/{}/centers_k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), 'wb') as f:
        pickle.dump(centers, f) # should be [200, 50]
    with open('./clusters/{}/{}/{}/dist_k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), 'wb') as f:
        pickle.dump(dist, f) # should be [460048, 200]


def load_data():

    with open('./clusters/{}/{}/per={}/classIndex_k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), mode="rb") as f:
        classIndex = pickle.load(f)
    with open('./clusters/{}/{}/per={}{}/dist_k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), mode="rb") as f:
        dist = pickle.load(f)
    with open("../data/embedding/BGL/iforest-train-ws200-per{}.pkl".format(percent), mode="rb") as f:
        (x_tr, ifAnomaly) = pickle.load(f)

    print("load!!!!!!!: ", len(x_tr))

    iftest = False
    def test():
        a_list = dist[0].tolist()
        a_min = min(a_list)
        min_index = a_list.index(min(a_list))
        print("mini distance: ", a_min)
        print("the mini class Index: ", min_index)
        print("the actual class index: ", classIndex[0])
    if iftest == True:
        test()

    info = []
    for i in range(len(classIndex)):
        info.append((i, classIndex[i], dist[i][classIndex[i]], ifAnomaly[i]))
    print(info[0])
    print(np.array(info).shape)

    # shuffle (same order) for the following
    SEED = 1
    random.seed(SEED)
    random.shuffle(info)

    # split labelled and unlabelled instances
    labelledNormalInstances = []
    labelledAnormalyInstances = []
    unlabelledNormalInstances = []
    unlabelledAnormalyInstances = []

    count = {}
    for key in ifAnomaly:
        count[key] = count.get(key, 0) + 1
    NormalNum = count[0]
    AnormalyNum = count[1]
    print("the number of normal and anomalous instances are: ", count)

    for index in range(len(info)):
        add_label = list(info[index])
        if (info[index][3] == 0):
            if (len(labelledNormalInstances) <= math.ceil(NormalNum/2)):
                add_label.append("labeled")
                labelledNormalInstances.append(add_label)
            else:
                add_label.append("unlabeled")
                unlabelledNormalInstances.append(add_label)
        if (info[index][3] == 1):
            if (len(labelledAnormalyInstances) <= math.ceil(AnormalyNum/2)):
                add_label.append("labeled")
                labelledAnormalyInstances.append(add_label)
            else:
                add_label.append("unlabeled")
                unlabelledAnormalyInstances.append(add_label)
    info_list_withLabels = labelledNormalInstances + unlabelledNormalInstances + labelledAnormalyInstances + unlabelledAnormalyInstances
    print("labeled anormaly: ", len(labelledAnormalyInstances))
    print(info_list_withLabels[0:10])
    print(np.array(info_list_withLabels).shape)

    with open('./clusters/{}/{}/per={}/id_clusterIndex_dist_ifAnomaly_ifLabeled_k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), 'wb') as f:
        pickle.dump(info_list_withLabels, f)


def calProbability():

    with open('./clusters/{}/{}/per={}/id_clusterIndex_dist_ifAnomaly_ifLabeled_k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), 'rb') as f:
        info_list_withLabels = pickle.load(f)

    # print(info_list_withLabels[0:10])
    print(len(info_list_withLabels))
    SumLabeledDist_N = {}
    SumLabeledCount_N = {}
    SumLabeledDist_A = {}
    SumLabeledCount_A = {}
    UnlabeledDist = {} # for each unlabeled sample, build a pair (index: distance)

    for i in range(len(info_list_withLabels)):

        classLabel = info_list_withLabels[i][1]
        if classLabel not in SumLabeledDist_N.keys():
            SumLabeledDist_N[classLabel] = 0
        if classLabel not in SumLabeledDist_A.keys():
            SumLabeledDist_A[classLabel] = 0
        if classLabel not in SumLabeledCount_N.keys():
            SumLabeledCount_N[classLabel] = 0
        if classLabel not in SumLabeledCount_A.keys():
            SumLabeledCount_A[classLabel] = 0

        if info_list_withLabels[i][4] == 'unlabeled':
            UnlabeledDist[info_list_withLabels[i][0]] = info_list_withLabels[i][2]

        if (info_list_withLabels[i][4] == 'labeled'):
            if (info_list_withLabels[i][3]==0):
                SumLabeledDist_N[classLabel] = SumLabeledDist_N[classLabel] + info_list_withLabels[i][2]
                SumLabeledCount_N[classLabel] = SumLabeledCount_N[classLabel] + 1
            else:
                SumLabeledDist_A[classLabel] = SumLabeledDist_A[classLabel] + info_list_withLabels[i][2]
                SumLabeledCount_A[classLabel] = SumLabeledCount_A[classLabel] + 1


    dist_N = {}
    dist_A = {}
    for key, value in SumLabeledDist_N.items():
        if SumLabeledCount_N.get(key):
            SumLabeledDist_N[key] = round(float(SumLabeledDist_N[key]) / SumLabeledCount_N[key], 2)
    for key, value in SumLabeledDist_A.items():
        if SumLabeledCount_A.get(key):
            SumLabeledDist_A[key] = round(float(SumLabeledDist_A[key]) / SumLabeledCount_A[key], 2)
    dist_N = SumLabeledDist_N
    dist_A = SumLabeledDist_A

    # print(dist_N)
    # print(dist_A)
    # print("For each cluster, the average distances of labeled normal and abnomalous samples: ")
    # print(dist_N)
    # print(len(dist_N))
    # print(dist_A)
    # print(len(dist_A))
    # print("the number of clusters with labeled instances: ", len(dist_Union))

 # probablity calculation for each sample in the training set.
    print("P(normal) for each sample in the training set.\n ")
    probabilityforNormal = {}

    # print("before for loop!!!!!!!!!!!!")
    # print(info_list_withLabels[0:10]) # has been shuffled

    for i in range(len(info_list_withLabels)):
        # print(info_list_withLabels[0:10])
        p = 0
        if (info_list_withLabels[i][4] == 'labeled'):
            if (info_list_withLabels[i][3] == 1): # anomaly
                probabilityforNormal[info_list_withLabels[i][0]] = 0
                # print("label case 1")
            else: # normal
                probabilityforNormal[info_list_withLabels[i][0]] = 1
                # print("label case 2")

        elif (info_list_withLabels[i][4] == 'unlabeled'):
            d_N = 0
            d_A = 0
            if (info_list_withLabels[i][1] in dist_N.keys()):
                d_N = dist_N.get(info_list_withLabels[i][1])
            else:
                d_N = 99999

            if (info_list_withLabels[i][1] in dist_A.keys()):
                d_A = dist_A.get(info_list_withLabels[i][1])
            else:
                d_A = 99999

            if ((d_N==99999) & (d_A==99999)):
                probabilityforNormal[info_list_withLabels[i][0]] = 0.5
                # print("unlabeled case 1")
            elif ((d_N==99999) & (d_A!=99999)): # only anomaly
                probabilityforNormal[info_list_withLabels[i][0]] = 0.01
                # print("unlabeled case 2")
            elif ((d_N!=99999) & (d_A==99999)): # only normal
                probabilityforNormal[info_list_withLabels[i][0]] = 0.99
                # print("unlabeled case 3")
            elif ((d_N!=99999) & (d_A!=99999)):
                d = info_list_withLabels[i][2] # no problem
                # case 1:
                if ((d < d_N) & (d_N < d_A)):
                    p = 1 - ((d_N - d)/(d_A - d)) * (d_N / d_A) * 0.5
                    # print("unlabeled case 4")
                # case 2:
                elif ((d_N <= d) & (d < d_A)):
                    if ((d-d_N) <= (d_A-d)):
                        p = 0.5 + (1 - (d-d_N)/(d_A-d)) * (1 - (d_N / d_A)) * 0.5
                        # print("unlabeled case 5")
                    else:
                        p = 0.5 - (1 - (d_A - d)/(d - d_N)) * (1 - (d_N / d_A)) * 0.5
                        # print("unlabeled case 6")
                # case 3:
                elif ((d_N < d_A) & (d_A<= d)):
                    p = 0.5 - (1 - (d - d_A)/(d - d_N)) * (1 - (d_N / d_A)) * 0.5
                    # print("unlabeled case 7")
                # case 4:
                elif ((d < d_A) & (d_A < d_N)):
                    p = 0 + ((d_A - d)/(d_N - d)) * (d_A / d_N) * 0.5
                    # print("unlabeled case 8")
                # case 5:
                elif ((d_A <= d) & (d < d_N)):
                    if ((d-d_A) <= (d_N-d)):
                        p = 0.5 - (1 - (d - d_A)/(d_N - d)) * (1 - (d_A / d_N)) * 0.5
                        # print("unlabeled case 9")
                    else:
                        p = 0.5 + (1 - (d_N - d)/(d - d_A)) * (1 - (d_A / d_N)) * 0.5
                        # print("unlabeled case 10")
                # case 6:
                elif ((d_A < d_N) & (d_N <= d)):
                    p = 0.5 + (1 - (d - d_N)/(d - d_A)) * (d_A / d_N) * 0.5
                    # print("unlabeled case 11")
                #case 7:
                elif (d_A == d_N):
                    p = 0.5
                    # print("unlabeled case 12")
                probabilityforNormal[info_list_withLabels[i][0]] = p

    # print(probabilityforNormal)
    # print(len(probabilityforNormal))
    # print("After the loop!!!!!!!!!!!!")
    # print(list(probabilityforNormal.items())[:10])

    with open('./clusters/{}/{}/per={}/probability/k={}_d={}.pickle'.format(dataset, ws, percent, k, n_components), 'wb') as f:
        pickle.dump(probabilityforNormal, f)

    def get_key (dict, value):
        i = 0
        for key, val in dict.items():
            if val == value:
                i = i + 1
        return i
    print(get_key(probabilityforNormal, 1))









if __name__ == '__main__':
    emb_dir = './data/embeddings/BGL'
    ws = 'ws=20'
    percent = 8 # The ratio of normal and abnormal
    # For BGL, the undersampling ratio are 8 (ws=20), 7 (ws=100), and 6 (ws=200) according to the undersampling rules (75%).
    dataset = 'BGL'
    k_list= [14, 46] # the obtained two optimal k values, i.e., 14 and 46 for ws=20 (BGL).
    n_components = 50
    for i in range(len(k_list)):
        k=k_list[i]
        kmeans()
        load_data()
        calProbability()

# Note: since shuffle for each situation (different k values) is the same
# so the order of the indexes keeps same for each situation.