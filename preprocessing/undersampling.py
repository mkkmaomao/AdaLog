import numpy as np
import pickle
from collections import Counter
def loadtrain(ws, percent):
    # with open("./HDFS/neural-train.pkl", mode="rb") as f:
    with open("../data/embedding/BGL/iforest-train-ws{}.pkl".format(ws), mode="rb") as f:
        (x_tr, y_tr) = pickle.load(f)
    x_sum_tr = []
    y_sum_tr = []
    countN = 0
    print(len(y_tr))
    # y_tr = y_tr.tolist()
    # the tolist line is for dataset HDFS.
    print(y_tr.count(1))
    for i in range(len(y_tr)-1, -1, -1):
        if (y_tr[i] == 0):
            if (countN < (y_tr.count(1)*percent)):
                x_sum_tr.append(x_tr[i])
                y_sum_tr.append(y_tr[i])
                countN = countN + 1
            else:
                pass
        else:
            x_sum_tr.append(x_tr[i])
            y_sum_tr.append(y_tr[i])

    print("after the 1st loop (A: {} and N: {})".format(y_sum_tr.count(1),y_sum_tr.count(0)))
    print("y_sum_tr: ",len(y_sum_tr))

    # with open("./HDFS_sampling/iforest-train-per{}.pkl".format(percent), mode="wb") as f:
    with open("../data/embedding/BGL/iforest-train-ws{}-per{}.pkl".format(ws, percent), mode="wb") as f:
        pickle.dump((x_sum_tr, y_sum_tr), f, protocol=pickle.HIGHEST_PROTOCOL)



# def loadtest(ws):
#     # with open("./HDFS/neural-test.pkl", mode="rb") as f:
#     with open("../data/embedding/BGL/iforest-test-ws{}.pkl".format(ws), mode="rb") as f:
#         (x_te, y_te) = pickle.load(f)
#     x_sum_te = []
#     y_sum_te = []
#     countA = 0
#     countN = 0
#     for i in range(len(y_te)):
#         if y_te[i] == 0:
#                 x_sum_te.append(x_te[i])
#                 y_sum_te.append(y_te[i])
#                 countN = countN + 1
#         else:
#             x_sum_te.append(x_te[i])
#             y_sum_te.append(y_te[i])
#             countA = countA + 1
#
#     print("the number of anomaly and normal samples in test set are {} and {}. ".format(countA, countN))
#     # with open("./HDFS_sampling/iforest-test.pkl".format(ws), mode="wb") as f:
#     with open("../data/embedding/BGL/iforest-test-ws{}.pkl".format(ws), mode="wb") as f:
#         pickle.dump((x_sum_te, y_sum_te), f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    p = [8] # for BGL, use the undersampling ratio is 8 according to the undersampling rules.
    for i in range(len(p)):
        per = p[i]
        loadtrain(ws=200,percent=per)
        # loadtest(ws=200)