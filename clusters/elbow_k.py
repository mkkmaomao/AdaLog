import os
import pickle
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def chooseclusternum(senVectors, dataset, a, b, per):
    K = range(a,b)
    meandistortions = []
    for k in K:
        clf = KMeans(n_clusters=k, max_iter=10000, init="k-means++", tol=1e-6)

        # 1. Elbow method
        clf.fit(senVectors)
        meandistortions.append(
            sum(np.min(cdist(senVectors,clf.cluster_centers_,'euclidean'), axis=1))
            /senVectors.shape[0])

    # plot for the elbow method
    print("meandistortions: ", meandistortions)
    plt.plot(K, meandistortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel(u'Average degree of distortion')
    plt.title(u'The elbow rule determines the best k value')
    path = r'./clusters/k_selection/{}'.format(dataset)
    print("PATH ", path)
    plt.show()


if __name__ == '__main__':
    emb_dir = './data/embedding/BGL'
    data = 'BGL'
    ws = 20
    # k=200
    a = 10
    b = 50
    n_components = 50
    percent = [7]
    for i in range(len(percent)):
        per = percent[i]
        # with open(os.path.join(emb_dir, "iforest-train-per{}.pkl".format(per)), mode="rb") as f:
        with open(os.path.join(emb_dir, "iforest-train-ws20-per{}.pkl".format(per)), mode="rb") as f:
            (x_tr, y_tr) = pickle.load(f)
        x_sum_tr = []
        for i in range(len(x_tr)):
            x_sum_tr.append(np.sum(x_tr[i], axis=0))
        x_sum_tr = np.array(x_sum_tr)
        # print(x_sum_tr.shape)
        # print(type(x_sum_tr))
        # print(len(y_tr))
        # print(y_tr[:10])


        pca = PCA(n_components=n_components)
        x = pca.fit_transform(x_sum_tr)

        k_opt = chooseclusternum(x, dataset=data, a=a, b=b, per=per)

        # with open(os.path.join(emb_dir, "neural-test.pkl"), mode="rb") as f:
        #     (x_te, y_te) = pickle.load(f)

        # centroids, clusterAssment = kMeansPlus(dataSet=x_sum_tr, k=k, d=n_components, distMeas=distEclud, createCent=initialize)
        # P_normal_cal(centroids, clusterAssment)