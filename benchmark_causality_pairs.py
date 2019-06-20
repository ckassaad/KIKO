import numpy as np
import tensorflow as tf
from ANM_pHSIC import ANM_ordering
from kiko import kiko
import time
E = []
time_list = []
cost_list = []
method = 'kiko'
nb_runs = 1
for i in range(0,nb_runs):
    print('--------------------- ITERATION '+str(i + 1)+'-------------------------')
    R = dict()
    P = []
    err=0
    kk=0
    A = [0]*46 + [1]*7 + [0] +[1]*9 + [0]*4 + [1]*2 +[0]*3 + [1,0,1,0,1,0,1,1] + [0]*3 +[1] + [0]*4 + [1,1,0,1] +[0]*6 + [1] +[0]*6 + [1,0,1]
    for k in range(1, 101):
        start = time.time()
        print(k)
        data = np.loadtxt('benchmark/pair' + str(k).zfill(4) + '.txt')
        data = data[:, :2]

        answer = A[k-1]
        if answer != 2:
            n_dim=data.shape[1]

            if method=='kiko':
                learning_rate = 0.01
                stddev = 0.1
                training_epoch = 300
                data_split = True
                activation_ih = tf.nn.tanh
                activation_ho = None
                # activation_ih = tf.nn.relu
                # activation_ho = tf.nn.tanh
                noise = True
                zeros = False
                stochastic =False
                cnn = False
                if cnn:
                    num_neurons = 16
                else:
                    num_neurons = 10
                alpha = 0.01
                res = kiko(data, learning_rate, training_epoch, num_neurons, stddev, activation_ih, activation_ho,
                             noise, zeros, stochastic, cnn, data_split, alpha=alpha)
                sig=res['order']
                pa = dict()
                for i in range(len(sig)):
                    pa[sig[i]] = list(sig[:i])

            elif method == 'anm':
                data_split = True
                res = ANM_ordering(data, method='gpr', data_max=3000, data_split=data_split)
                sig = res['order']
                pa = dict()
                for i in range(len(sig)):
                    pa[sig[i]] = list(sig[:i])

            else:
                print('unavailable method')
                exit(0)

            if pa[0] == [1] and pa[1] == []:
                pred = 1
            elif pa[0] == [] and pa[1] == [0]:
                pred = 0
            else:
                pred = 2
            R[k] = [answer, pred]
            P.append(pred)
            print(R[k])
            if answer!=2:
                kk=kk+1
                err=err+abs(pred-answer)
                print(err/kk)

            res1 = list(R.values())
            res2 = [np.abs(x[0] - x[1]) for x in res1 if (x[0] != 2 and x[0] != 2)]
            res3 = np.sum(res2) / len(res2)
        else:
            print('this dataset will not be treated')
        end = time.time()
        time_list.append(end-start)
        cost_list.append(res['cost_history'])
    E.append(res3)
    print('Errors = '+str(E))
    accuracy = 1-np.mean(E)

print("-------------------SETTINGS----------------------")
print('num_neurons :'+str(num_neurons))
print('activation_hidden: '+str(activation_ih))
print('activation_io: '+str(activation_ho))
print('learning_rate: '+str(learning_rate))
print('stddev: '+str(stddev))
print('training_epoch: '+str(training_epoch))
print('noise: '+str(noise))
print('zeros: '+str(zeros))
print('stochastic :'+str(stochastic))
print('data_split: '+str(data_split))
print('cnn architecture: ' + str(cnn))
print('nb runs: ' + str(nb_runs))
print()
print("---------------------RESULTS-------------------------")
print('time mean: ' + str(np.mean(time_list)))
print('total training cost mean: '+ str(np.mean(cost_list)))
print('accuracy :'+str(accuracy))