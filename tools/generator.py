#import libraries
import numpy as np


# simulated data
def data_sim4(N=1000):
    E1 = np.random.uniform(low=-1.0, high=1.0, size=N)
    E2 = np.random.uniform(low=-1.0, high=1.0, size=N)
    E3 = np.random.uniform(low=-3.0, high=3.0, size=N)
    E4 = np.random.uniform(low=-1.0, high=1.0, size=N)

    X3 = E3
    X2 = np.power(X3, 2) + E2
    X4 = 4 * np.sqrt(np.abs(X3)) + E4
    X1 = 2 * np.sin(X2) + 2 * np.sin(X4) + E1

    data = np.column_stack((X1, X2, X3, X4))
    return data


def data_sim_n(N=1000, n=20):
    return np.random.normal(loc=0.0, scale=1.0, size=[N,n])



