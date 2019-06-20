# -*- coding: utf-8 -*-
"""
Created on Wed Feb 14 11:35:42 2018

@author: karim assaad
"""
import numpy as np
from joblib import Parallel, delayed

class HSIC:
    def __init__(self, kernel='rbf', gamma='median', parallel = True, median_data=1000):
        """
        :param kernel: rbf or laplace
        :param gamma: If overestimated, the exponential will behave almost linearly and the higher-dimensional
        projection will start to lose its non-linear power. In the other hand, if underestimated, the function will
         lack regularization and the decision boundary will be highly sensitive to noise in training data
         by debault gamma is equal to the median distance.
        """
        kernel_name = {
            'rbf': rbf,
            'laplace': laplace
        }

        if isinstance(kernel, list):
            if len(kernel) == 1:
                self.kernel1 = kernel_name[kernel[0]]
                self.kernel2 = kernel_name[kernel[0]]
            else:
                self.kernel1 = kernel_name[kernel[0]]
                self.kernel2 = kernel_name[kernel[1]]
        else:
            self.kernel1 = self.kernel2 = kernel_name[kernel]

        self.gamma = gamma
        self.k1 = 0
        self.k2 = 0
        self.parallel = parallel
        self.median_data = median_data

    def fit(self, x, y):
        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)

        if self.gamma == 'median':
            gamma1 = median_distance_fast(x, self.median_data)
            gamma2 = median_distance_fast(y, self.median_data)
            self.gamma = [gamma1, gamma2]
        else:
            gamma1 = self.gamma[0]
            gamma2 = self.gamma[1]

        m = x.shape[0]
        k1 = self.kernel1(x, x, sigma=gamma1)
        k2 = self.kernel2(y, y, sigma=gamma2)

        self.k1 = k1
        self.k2 = k2

        temp1 = (k1*k2).sum()
        k1 = k1.sum(axis=1)
        k2 = k2.sum(axis=1)
        temp2 = ((1/m)**2)*k1.sum()*k2.sum()
        temp3 = (2.0/m)*np.dot(k1, k2)
        hsic_value = (temp1+temp2-temp3)*(1/m)**2
        return hsic_value


def median_distance(x, median_data=1000, parallel=True, num_processor = 4):
    n = x.shape[0]
    # d=x.shape[1]
    if n > median_data:
        n = median_data
        x = x[np.random.randint(x.shape[0], size=n), :]
    lentot = n*(n+1)/2-n
    middle = lentot/2
    distance_list = []
    # xnorm = 0.0
    if parallel:
        # TO BE COMPLETED (parallisation of the above loop)
        # num_processor = 4
        try:
            distance_list = Parallel(n_jobs=num_processor)(delayed(xnom_func)(i, x) for i in range(n))
            distance_list = [i for sub in distance_list for i in sub]
        except:
            print('Problem with parallelisation of the median distance in HSIC. Try setting the parameter parallel to False')
    else:
        for i in range(n):
            # for j in range((i+1),n):
            #     for l in range(d):
            #         xnorm += np.sum(np.power(x[i, l]- x[j, l], 2))
            #     bandvec.append(xnorm)
            #     xnorm = 0.0
            xnorm = np.sum(np.power(np.subtract(x[i, :], x[(i+1):, :]), 2), axis=1)
            distance_list = distance_list+list(xnorm)

    v = np.sort(distance_list)
    median = v[int(middle)]
    median = np.sqrt(median*0.5)

    if median == 0:
        median = 0.001
    return median


def median_distance_fast(y, median_data=1000):
    n = y.shape[0]
    if n > median_data:
        n = median_data
        y = y[np.random.randint(y.shape[0], size=n), :]
    G = np.sum(y*y, 1).reshape(n,1)
    Q = np.tile(G, (1, n) )
    R = np.tile(G.T, (n, 1) )

    dists = Q + R - 2* np.dot(y, y.T)
    dists = dists - np.tril(dists)
    dists = dists.reshape(n**2, 1)

    width_y = np.sqrt( 0.5 * np.median(dists[dists>0]) )
    return width_y


def xnom_func(i, x):
    return np.sum(np.power(np.subtract(x[i, :], x[(i+1):, :]), 2), axis=1)


def rbf(x, y, sigma=1):
    # k = np.power(np.sqrt(np.power(x[:, :, None] - y[:, :, None].T, 2)), 2)
    k = np.power(np.sqrt(np.power(np.subtract(x[:, :, None], y[:, :, None].T), 2)), 2)
    k = np.exp(- np.divide(k.sum(1), 2 * (sigma ** 2)))
    return k


def laplace(x, y, sigma=1):
    k = np.sqrt(np.power(x[:, :, None] - y[:, :, None].T, 2))
    # k = np.exp(-gamma * k.sum(1))
    k = np.exp(- np.divide(k.sum(1), sigma))
    return k


if __name__ == "__main__":
    np.random.seed(10)

    A = np.random.uniform(0, 1, 300)

    B = A*2 -0.1
    A = A.reshape(-1, 1)
    B = B.reshape(-1, 1)

    import time

    start = time.time()
    hs = HSIC(kernel='rbf')
    print(hs.fit(A, B))
    end = time.time()
    print(end - start)

