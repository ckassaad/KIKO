# -*- coding: utf-8 -*-
"""

Created on Wed Feb 07 10:45:27 2018

@author: Karim Assaad

Python 3.6
"""

import tensorflow as tf
import numpy as np
from tools.generator import data_sim4, data_sim_n
from tools.hsic import HSIC
# from tools.test_hsic import TestHSIC as HSIC
from sklearn.preprocessing import MinMaxScaler
import time


def feed_forward(x, weights, biases, activation_one, activation_two, i=0):
    """
    :param x: input
    :param weights: weights associated with the neural network
    :param biases: biases associated with the neural network
    :param activation_one: activation function for the first layer
    :param activation_two: activation function for the last layer
    :param i: number assioiated to the archetecture
    :return: values assiociated to the output layer
    """
    first = 'h'+str(2*i)
    second = 'h' + str(2*i + 1)
    layer = x
    layer = tf.add(tf.matmul(layer, weights[first]), biases[first])
    layer = activation_one(layer)
    out = tf.add(tf.matmul(layer, weights[second]), biases[second])
    if activation_two is not None:
        out = activation_two(out)
    return out, layer


# Function to produce noise
def add_noise(x, beta=0.5):
    x = x.copy()
    rand = np.random.randint(0, high=x.shape[1], size=x.shape[0])
    for i in range(x.shape[0]):
        proba = np.random.random(size=1)
        if proba > beta:
            x[i, rand[i]] = 0
    return x


def kiko(data, learning_rate, training_epoch, num_neurons, stddev, activation_one, activation_two, noise,
         stochastic, data_split=True, data_max=3000, alpha=0.01):
    """
    :param data: input
    :param learning_rate: learning rate of the autoencoder
    :param training_epoch: number of training epochs
    :param num_neurons: number of neurones in the hidden layer
    :param stddev: std for weight Initialization
    :param activation_one: first activation function
    :param activation_two:  second activation function
    :param noise: boolean value, if true a denoising autoencoder should be use
    :param stochastic: boolean value
    :param data_split: boolean value
    :param alpha:
    :return: dict
    """
    # causal ordering
    start = time.time()
    tf.reset_default_graph()
    if data.shape[0] > data_max:
        idx = np.random.permutation(data.shape[0])[:data_max]
        data = data[idx, :]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    data = scaler.fit_transform(data)

    m = data.shape[1]
    x_holder = dict()
    y_holder = dict()
    weights = dict()
    biases = dict()

    for i in range(m):
        x_holder[i] = tf.placeholder(tf.float32, shape=[None, m-i])
        y_holder[i] = tf.placeholder(tf.float32, shape=[None, m-i])
        weights['h' + str(2*i)] = tf.Variable(tf.truncated_normal([m-i, num_neurons], stddev=stddev), name='W_'+str(2*i))
        biases['h' + str(2*i)] = tf.Variable(tf.truncated_normal([num_neurons], stddev=stddev), name='b_'+str(2*i))
        weights['h' + str(2*i+1)] = tf.Variable(tf.truncated_normal([num_neurons, m-i], stddev=stddev), name='W_'+str(2*i+1))
        biases['h' + str(2*i+1)] = tf.Variable(tf.truncated_normal([m-i], stddev=stddev), name='b_'+str(2*i+1))

    out = dict()
    hlayer = dict()
    J = dict()
    error = dict()
    optimizer = dict()
    cost_history = []
    hsic_history = []

    if data_split:
        split = int(data.shape[0]/2)
        np.random.shuffle(data)
        train, test = data[:split, :], data[split:, :]
    else:
        train = data.copy()
        test = data.copy()

    for i in range(m):
        out[i], hlayer[i] = feed_forward(x_holder[i], weights, biases, activation_one, activation_two, i=i)

        J[i] = tf.losses.mean_squared_error(out[i], y_holder[i])
        error[i] = tf.subtract(y_holder[i], out[i])
        optimizer[i] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(J[i])

    S = list(range(m))
    sig = np.zeros(m, dtype=int)

    for j in range(m-1):
        if j != 0:
            train = np.delete(train, [idp], axis=1)
            test = np.delete(test, [idp], axis=1)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for step in range(training_epoch):
                if stochastic:
                    mini_batch_size = 1
                    N = train.shape[0]
                    n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
                    i_batch = (step % n_batch) * mini_batch_size
                    if noise:
                        train_x = add_noise(train[i_batch:i_batch + mini_batch_size])
                    else:
                        train_x = train[i_batch:i_batch + mini_batch_size].copy()
                    batch = train_x, train[i_batch:i_batch + mini_batch_size]
                    feed = {x_holder[j]: batch[0], y_holder[j]: batch[1]}
                else:
                    if noise:
                        train_x = add_noise(train)
                    else:
                        train_x = train.copy()
                    feed = {x_holder[j]: train_x, y_holder[j]: train}
                cost = sess.run(J[j], feed_dict=feed)
                sess.run(optimizer[j], feed_dict=feed)
            cost_history.append(cost)
            print('epoch: ', step, ' - ', 'cost: ', "{:.4f}".format(cost, 4))

            k = m - 1 - j
            hsic_values = []
            print(len(S))
            for i in range(len(S)):
                test_temp = test.copy()
                test_temp[:, i] = np.zeros((test.shape[0]))
                feed = {x_holder[j]: test_temp, y_holder[j]: test}
                e = sess.run(error[j], feed_dict=feed)[:, i]
                feed = {x_holder[j]: test}
                c = sess.run(hlayer[j], feed_dict=feed)
                hs = HSIC(kernel='rbf')
                hsic_values.append(hs.fit(c, e))

            print('hsics :'+str(hsic_values))
            hsic_history.append(hsic_values)
            idp = np.argmin(hsic_values)
            sig[k] = S[idp]
            del S[idp]
            print(sig)

    if len(S) == 1:
        sig[0] = S[0]
    print(sig)

    end = time.time()
    order_time = end - start
    print('time causal ordering: '+str(order_time))
    noise = False
    stochastic = False
    # create dict
    pa = dict()
    for i in range(len(sig)):
        pa[sig[i]] = list(sig[:i])

    p_list_discovery = []
    # causal discovery
    start = time.time()
    tf.reset_default_graph()
    x_holder = dict()
    weights = dict()
    biases = dict()

    y_1 = tf.placeholder(tf.float32, shape=[None, 1])
    for i in range(1, m):
        x_holder[i] = tf.placeholder(tf.float32, shape=[None, m-i])
        weights['h' + str(2*i)] = tf.Variable(tf.truncated_normal([m-i, num_neurons], stddev=stddev), name='W_'+str(2*i))
        biases['h' + str(2*i)] = tf.Variable(tf.truncated_normal([num_neurons], stddev=stddev), name='b_'+str(2*i))
        weights['h' + str(2*i+1)] = tf.Variable(tf.truncated_normal([num_neurons, 1], stddev=stddev), name='W_'+str(2*i+1))
        biases['h' + str(2*i+1)] = tf.Variable(tf.truncated_normal([1], stddev=stddev), name='b_'+str(2*i+1))

    out = dict()
    hlayer = dict()
    J = dict()
    error = dict()
    optimizer = dict()
    cost_history_discovery = []
    hsic_history_discovery = []
    discovery_time = 0
    if m>2:
        for i in range(1, m):
            out[i], hlayer[i] = feed_forward(x_holder[i], weights, biases, activation_one, activation_two, i=i)
            J[i] = tf.losses.mean_squared_error(out[i], y_1)
            error[i] = tf.subtract(y_1, out[i])
            optimizer[i] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(J[i])
        for j in reversed(sig):
            if len(pa[j]) > 0:
                nb = m-len(pa[j])
                train_x, test_x = data[:split, pa[j]], data[split:, pa[j]]
                train_y, test_y = data[:split, j].reshape(-1, 1), data[split:, j].reshape(-1, 1)
                init = tf.global_variables_initializer()
                with tf.Session() as sess:
                    sess.run(init)
                    for step in range(training_epoch):
                        if stochastic:
                            mini_batch_size = 1
                            N = train.shape[0]
                            n_batch = N // mini_batch_size + (N % mini_batch_size != 0)
                            i_batch = (step % n_batch) * mini_batch_size
                            if noise:
                                train_x = add_noise(train[i_batch:i_batch + mini_batch_size])
                            else:
                                train_x = train_x[i_batch:i_batch + mini_batch_size]
                            batch = train_x, train_y[i_batch:i_batch + mini_batch_size]
                            feed = {x_holder[nb]: batch[0], y_1: batch[1]}
                        else:
                            if noise:
                                train_x = add_noise(train_x)
                            feed = {x_holder[nb]: train_x, y_1: train_y}
                        cost = sess.run(J[nb], feed_dict=feed)
                        sess.run(optimizer[nb], feed_dict=feed)
                    cost_history_discovery.append(cost)
                    print('epoch: ', step, ' - ', 'cost: ', "{:.4f}".format(cost, 4))

                    pa_temp = pa[j].copy()
                    print(pa_temp)
                    for p in pa_temp:
                        # print('parent: '+str(p))
                        test_temp = test_x.copy()
                        idp = int(np.argwhere(pa_temp == p).reshape(1))
                        test_temp[:, idp] = np.zeros(test_x.shape[0])
                        feed = {x_holder[nb]: test_temp, y_1: test_y}
                        e = sess.run(error[nb], feed_dict=feed)
                        feed = {x_holder[nb]: test_x}
                        c = sess.run(hlayer[nb], feed_dict=feed)
                        hs = HSIC(kernel='rbf')
                        hsic_value = hs.fit(c, e)
                        hsic_history_discovery.append(hsic_value)

        # cut unneccessary arrows
        list_hsic = np.array(hsic_history_discovery)
        sort_index = np.argsort(list_hsic)
        sort_list_hsic = list_hsic[sort_index]
        threshold = np.inf
        threshold_list = []
        val_init = sort_list_hsic[0]
        for val_ind in sort_list_hsic[1:]:
            threshold_new = val_ind - val_init
            if threshold_new > 2*threshold:
                alpha = val_init
                print('alpha = '+str(alpha))
                break
            else:
                threshold = threshold_new
            val_init = val_ind
            threshold_list.append(threshold_new)

        if len(threshold_list) < (len(sort_index)-1):
            seuil_iterator = 0
            for j in reversed(sig):
                pa_temp = pa[j].copy()
                for p in pa_temp:
                    if seuil_iterator in sort_index[:(len(threshold_list)+1)]:
                        pa[j].remove(p)
                    seuil_iterator = seuil_iterator+1

        end = time.time()
        discovery_time = end - start
        print('time causal discovery: '+str(discovery_time))

    result = {'order': sig,
              'order time': order_time,
              'cost_history': cost_history,
              'hsic_history': hsic_history,
              'discovery': pa,
              'discovery_time': discovery_time,
              'cost_history_discovery': cost_history_discovery,
              'hsic_history_discovery': hsic_history_discovery,
              'p_history_discovery' : p_list_discovery,
              'alpha': alpha
              }
    return result


def hamming_distance(d1, d2):
    m1 = np.zeros([n, n])
    m2 = np.zeros([n, n])
    for j in range(n):
        m1[j, d1[j]] = 1
        m2[j, d2[j]] = 1
    non_diag = np.where(~np.eye(m1.shape[0], dtype=bool))
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
    nb_runs = 100

    learning_rate = 0.01
    stddev = 0.1
    training_epoch = 300
    data_split = True
    activation_ih = tf.nn.relu
    activation_ho = tf.nn.tanh
    # activation_ih = tf.nn.tanh
    # activation_ho = None
    noise = True

    stochastic = False
    num_neurons = 10
    alpha = 0.01
    # ###########################################################

    time_list = []
    time_list_discovery = []
    order_array= np.zeros([n,nb_runs], dtype=int)
    cost_array = []
    parent_list = []
    hamming_list=[]


    for i in range(nb_runs):
        print('---------------------     iteration '+str(i)+'    -------------------')
        if n == 4:
            data = data_sim4(N=500)
            true_order = [[2, 1, 3, 0], [2, 3, 1, 0]]
            true_structure = {0:[1,3], 1: [2], 2:[], 3: [2]}
        else:
            data = data_sim_n(N, n)
            true_order = None
            true_structure = None

        res = kiko(data, learning_rate, training_epoch, num_neurons, stddev, activation_ih, activation_ho, noise,
                         stochastic, data_split, data_max=N, alpha=alpha)
        print(res)
        order_array[:, i] = res['order']
        cost_array.append(res['cost_history'])
        time_list.append(res['order time'])
        time_list_discovery.append(res['discovery_time'])
        if true_structure:
            hamming_list.append(hamming_distance(true_structure, res['discovery']))
        parent_list.append(str(res['discovery']))

    cost_array = np.array(cost_array)

    accuracy = 0
    order_nodes = []
    proba_nodes = []
    order_sequence = []
    proba_sequence = []

    from collections import Counter
    if true_order:
        for i in range(nb_runs):
            accuracy += (list(order_array[:, i]) in true_order)
        accuracy = accuracy/nb_runs

        for i in range(n):
            order_nodes.append(np.bincount(order_array[i,:]).argmax())
        stability_nodes = (order_array.transpose() == order_nodes).all(-1).sum()/nb_runs

        for i in range(len(order_nodes)):
            proba_nodes.append(list(order_array[i, :]).count(order_nodes[i])/nb_runs)

        order_tuple = map(tuple, order_array.transpose())  # must convert to tuple because list is an unhaable type
        final_count = Counter(order_tuple)
        order_sequence = final_count.most_common(1)[0][0]
        stability_sequence = final_count.most_common(1)[0][1]/nb_runs

        for i in range(len(order_sequence)):
            proba_sequence.append(list(order_array[i,:]).count(order_sequence[i])/nb_runs)

    parent_count = Counter(parent_list)

    print("-------------------SETTINGS----------------------")
    print('num_neurons :'+str(num_neurons))
    print('activation_hidden: '+str(activation_ih))
    print('activation_io: '+str(activation_ho))
    print('learning_rate: '+str(learning_rate))
    print('stddev: '+str(stddev))
    print('training_epoch: '+str(training_epoch))
    print('noise: '+str(noise))
    print('stochastic :'+str(stochastic))
    print('data_split: '+str(data_split))
    print('nb runs: ' + str(nb_runs))
    print('alpha: ' + str(alpha))
    print()
    print("---------------------RESULTS-------------------------")
    print("---------------------Order-------------------------")
    print('time mean: ' + str(np.mean(time_list)))
    print('training cost mean for each NN: '+ str(cost_array.mean(axis=0)))
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
    print('alpĥa: ' + str(alpha))
    print('time mean: ' + str(np.mean(time_list_discovery)))
    print('parents list: ' + str(parent_count))
    if true_structure:
        print('hamming list ' + str(hamming_list))
        print('hamming mean: ' + str(np.mean(hamming_list)))
    print('final time: ' + str(np.mean(time_list) + np.mean(time_list_discovery)))



