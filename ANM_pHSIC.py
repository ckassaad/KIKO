import numpy as np
from tools.test_hsic import TestHSIC as HSIC
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.linear_model import Ridge as LSR
from sklearn.metrics import mean_squared_error as rms
from sklearn.preprocessing import MinMaxScaler

from tools.generator import data_sim4, data_sim_n
import time

def ANM_ordering(data, method='gpr', data_max=1000, data_split=True):
    start = time.time()

    if data.shape[0] > data_max:
        idx = np.random.permutation(data.shape[0])[:data_max]
        data = data[idx, :]
    print(data.shape)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)

    if data_split:
        split = int(data.shape[0]/2)
        np.random.shuffle(data)
        train, test = data[:split, :], data[split:, :]
    else:
        train = data.copy()
        test = data.copy()
    print(train.shape)
    print(test.shape)

    train_backup = train.copy()
    test_backup = test.copy()

    hsic_history = []
    cost_history = []
    n = data.shape[1]
    j = n
    S = list(range(n))
    sig = np.zeros(n, dtype=int)

    for k in range(n):
        print('iteration' + str(k))
        pvalues_values = np.array([])
        hsic_values = np.array([])
        if len(S) == 1:
            j = j - 1
            sig[j] = S[0]
        else:
            for i in range(len(S)):
                hsic = HSIC(method='gamma')
                if method == 'gpr':
                    model = GPR(normalize_y=False)
                    # model = GPR()
                elif method == 'lsr':
                    model = LSR(alpha=1.0)
                else:
                    print('error')
                    exit(0)

                print(np.delete(train, [i], axis=1).shape)
                model.fit(np.delete(train, [i], axis=1), train[:, i])
                rms_error = rms(train[:, i], model.predict(np.delete(train, [i], axis=1)))
                # Error of the prediction of variable i without the use of the input weights associated to variable i
                err = test[:, i] - model.predict(np.delete(test, [i], axis=1))
                print('cost: '+str(rms_error))
                cost_history.append(rms_error)
                # Measure of independence
                pvalues_values = np.append(pvalues_values, hsic.fit(np.delete(test, [i], axis=1), err))
                hsic_values = np.append(hsic_values, hsic.init_hsic)
                print(hsic_values)
                # Reset the weights of variable i to its initial values using the backup (BlackIn)
            j = j - 1
            print('pvalues ' + str(pvalues_values))
            print('hsics ' + str(hsic_values))
            idp = np.argmax(pvalues_values)
            if len([x for x in pvalues_values if x == np.max(pvalues_values)]) > 1:
                # if len(p[p == p[idp]]) > 1:
                hsic_values[pvalues_values != pvalues_values[idp]] = 1
                idp = np.argmin(hsic_values)

            sig[j] = S[idp]
            del S[idp]
            print(sig)
            hsic_history.append(hsic_values)
            train = np.delete(train, [idp], axis=1)
            test = np.delete(test, [idp], axis=1)
    # sig = np.array([2, 1, 3, 0])
    end = time.time()
    print('time ANM_ordering: '+str(end - start))
    result = {'order': sig,
              'order time': end - start,
              'cost_history': cost_history,
              'hsic_history': hsic_history,
              }

# def ANM_discovery(data, sig, alpha, method='gpr'):
    start = time.time()

    parent = dict()
    for i in range(len(sig)):
        parent[sig[i]] = list(sig[:i])

    hsic_list_discovery = []
    p_list_discovery = []
    for i in parent.keys():
        print('i: ' + str(i))
        pa = list(parent[i])
        train_temp = train_backup[:, pa].copy()
        test_temp = test_backup[:, pa].copy()
        print('pa: ' + str(pa))
        if len(pa)>1:
            for k in pa:
                print(k)
                hsic = HSIC(method='gamma')
                if method == 'gpr':
                    model = GPR(normalize_y=False)
                elif method == 'lsr':
                    model = LSR(alpha=1.0)
                else:
                    print('error')
                idk = int(np.argwhere(pa == k).reshape(1))
                print(idk)

                model.fit(np.delete(train_temp, [idk], axis=1), train_backup[:, i])
                rms_error = rms(train_backup[:, i], model.predict(np.delete(train_temp, [idk], axis=1)))
                err = test_backup[:, i] - model.predict(np.delete(test_temp, [idk], axis=1))

                print('cost: '+str(rms_error))
                p = hsic.fit(np.delete(test_temp, [idk], axis=1), err)
                # p = 1
                print('p-value: '+str(p))
                print('hsic: '+str(hsic.init_hsic))
                hsic_list_discovery.append(hsic.init_hsic)
                p_list_discovery.append(p)
                if (p >= alpha):
                    parent[i].remove(k)

            # p = hsic.fit(data[:, k], err)
            # if (p < alpha):
            #     parent_disc[i].remove(k)
    end = time.time()
    print('time ANM_discovery: ' + str(end - start))

    result['discovery'] = parent
    result['discovery_time'] = end - start
    result['hsic_history_discovery'] = hsic_list_discovery
    result['p_history_discovery'] = p_list_discovery
    return result


def hamming_distance(d1, d2):
    m1 = np.zeros([n, n])
    m2 = np.zeros([n, n])
    for j in range(n):
        m1[j, d1[j]] = 1
        m2[j, d2[j]] = 1
    non_diag = np.where(~np.eye(m1.shape[0], dtype=bool))
    print( " 6 = "+str(len(non_diag[0])))
    return (m1[non_diag] == m2[non_diag])\
               .sum()/len(non_diag[0])

if __name__ == "__main__":
    import sys

    if len(sys.argv)>1:
        n = int(sys.argv[1])
        if len(sys.argv)>2:
            N = int(sys.argv[2])
        print('Argument List:', str(sys.argv))
    else:
        n = 4
        N = 500



    # ############## set hyperparameters #######################
    nb_runs = 1

    data_split = True

    alpha = 0.01
    # ###########################################################

    time_list = []
    time_list_discovery = []
    order_array= np.zeros([n,nb_runs], dtype=int)
    cost_array = []
    parent_list = []
    hamming_list=[]

    dict_edges1 = {'0-2':0, '0-1':0, '0-3':0, '3-2':0, '3-1':0, '1-2':0}
    dict_edges2 = {'0-2': 0, '0-1': 0, '0-3': 0, '3-2': 0, '3-1': 0, '1-2': 0}
    dict_edges3 = {'0-2': 0, '0-1': 0, '0-3': 0, '3-2': 0, '3-1': 0, '1-2': 0}
    dict_edges4 = {'0-2': 0, '0-1': 0, '0-3': 0, '3-2': 0, '3-1': 0, '1-2': 0}
    dict_edges5 = {'0-2': 0, '0-1': 0, '0-3': 0, '3-2': 0, '3-1': 0, '1-2': 0}

    for i in range(nb_runs):
        print('---------------------     iteration '+str(i)+'    -------------------')
        if n == 4:
            data = data_sim4(N=500)
            true_order = [[2, 1, 3, 0], [2, 3, 1, 0]]
            true_structure = {0:[1,3], 1: [2], 2:[], 3: [2]}
        else:
            data = data_sim_n(N=N, n=n)
            true_order = None
            true_structure = None

        res = ANM_ordering(data, method='gpr', data_max=N, data_split=data_split)
        print(res)
        order_array[:, i] = res['order']
        cost_array.append(res['cost_history'])
        time_list.append(res['order time'])
        time_list_discovery.append(res['discovery_time'])
        if true_structure:
            hamming_list.append(hamming_distance(true_structure, res['discovery']))
        parent_list.append(str(res['discovery']))

    cost_array = np.array(cost_array)


    print(order_array)

    accuracy = 0
    order_nodes = []
    proba_nodes = []
    order_sequence = []
    proba_sequence = []

    from collections import Counter
    if true_order:
        for i in range(nb_runs):
            accuracy += (list(order_array[:,i]) in true_order)
        accuracy = accuracy/nb_runs

        for i in range(n):
            order_nodes.append(np.bincount(order_array[i,:]).argmax())
        stability_nodes = (order_array.transpose() == order_nodes).all(-1).sum()/nb_runs

        for i in range(len(order_nodes)):
            proba_nodes.append(list(order_array[i,:]).count(order_nodes[i])/nb_runs)

        order_tuple = map(tuple, order_array.transpose())  # must convert to tuple because list is an unhaable type
        final_count = Counter(order_tuple)
        order_sequence = final_count.most_common(1)[0][0]
        stability_sequence = final_count.most_common(1)[0][1]/nb_runs

        for i in range(len(order_sequence)):
            proba_sequence.append(list(order_array[i,:]).count(order_sequence[i])/nb_runs)

    parent_count = Counter(parent_list)

    print("-------------------SETTINGS----------------------")
    print('nb runs: ' + str(nb_runs))
    print('alpha: '+str(alpha))
    print()
    print("---------------------RESULTS-------------------------")
    print("---------------------Order-------------------------")
    print('time mean: ' + str(np.mean(time_list)))
    print('training cost mean for each model: '+ str(cost_array.mean(axis=0)))
    print('total training cost mean: ' + str(cost_array.mean()))
    if true_order:
        print('accuracy :'+str(accuracy))
        print("----------------by node---------------------")
        print('order by vote on nodes: '+str(order_nodes))
        print('stability by vote on nodes: ' + str(stability_nodes))
        print('probability by vote on nodes: ' + str(proba_sequence))
        print("----------------by sequence---------------------")
        print('order by vote on sequences: ' + str(order_sequence))
        print('stability of order by vote on sequences: '+str(stability_sequence))
        print('probability by vote on sequences: ' + str(proba_sequence))
        print()
    print("---------------------Discovery-------------------------")
    print('alpĥa: '+str(alpha))
    print('time mean: '+str(np.mean(time_list_discovery)))
    print('parents list: ' + str(parent_count))
    if true_structure:
        print('hamming list '+str(hamming_list))
        print('hamming mean: ' + str(np.mean(hamming_list)))
    print('final time: ' + str(np.mean(time_list) + np.mean(time_list_discovery)))