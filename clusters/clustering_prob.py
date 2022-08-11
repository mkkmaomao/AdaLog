import pickle

def load_probability(numofCluster, dim):
    with open('./clusters/{}/{}/per={}/probability/k={}_d={}.pickle'.format(dataset, ws,percent, numofCluster, dim), 'rb') as f:
        data = pickle.load(f)
        print(data)
    return data

def load_trueLabels():
    with open('../data/embeddings/{}/iforest-train-ws100-per{}.pkl'.format(dataset, percent), 'rb') as f:
        data = pickle.load(f)
        print(data)
    return data

def merge_dict(dict1, dict2):
    merge =  {}
    for key, value in dict1.items():
        if key in dict2.keys():
            merge[key] = dict2[key] + value
    return merge

def avg_dict(dict, dictNum):
    avg = {}
    for key, value in dict.items():
        avg[key] = value/dictNum
    return avg

def classification(ddict):
    estimate = {}
    trueLabel = {}
    TP, TN, FN, FP = 0, 0, 0, 0
    with open('./clusters/{}/{}/per={}/id_clusterIndex_dist_ifAnomaly_ifLabeled_k={}_d={}.pickle'.format(dataset, ws, percent, k1, 50),'rb') as f:
        info_list_withLabels = pickle.load(f)

    for i in range(len(info_list_withLabels)):
        id = info_list_withLabels[i][0]
        trueLabel[id] = info_list_withLabels[i][3]
        if (info_list_withLabels[i][4] == 'labeled'):
            estimate[id] = info_list_withLabels[i][3]
        else:
            # prob = dict.get(id)
            prob = ddict.get(id)
            if (prob>=0.50):
                estimate[id] = 0 # normal
            else:
                estimate[id] = 1 # abnomaly

    for key, value in trueLabel.items():
        if (key in estimate.keys()):
            if ((estimate[key] == value) & (value == 1)):
                TP = TP + 1
            elif ((estimate[key] == 1) & (value == 0)):
                FP = FP + 1
            elif ((estimate[key] == 0) & (value == 1)):
                FN = FN + 1

    Recall = TP / (TP + FN)
    Precision = TP / (TP + FP)
    F1 = 2 * Precision * Recall / (Precision + Recall)
    print("tp: ", TP)
    print("fn: ", FN)
    print("fp: ", FP)
    print("Recall: ", Recall )
    print("Precision: ", Precision)
    print("F1: ", F1)




if __name__ == "__main__":
    dataset = 'BGL'
    ws = 'ws=20'
    percent = 8
    k1 = 14
    k2 = 46
    d = 2
    n_components = 50
    dim_1 = load_probability(numofCluster=k1, dim=n_components)
    dim_2 = load_probability(numofCluster=k2, dim=n_components)

    dim_1 = avg_dict(dim_1, dictNum=d)
    dim_2 = avg_dict(dim_2, dictNum=d)

    merge_1 = merge_dict(dim_1,dim_2)

    with open('./clusters/{}/{}/per={}/probability/k={}&{}_d={}.pickle'.format(dataset, ws, percent, k1, k2, n_components), 'wb') as f:
        pickle.dump(merge_1, f)

    classification(merge_1)