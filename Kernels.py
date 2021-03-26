import numpy as np
from tqdm import tqdm


class Kernel:
    """
    Class that defines different types of kernels
    """

    def __init__(self, kernel="spectrum", gamma=0.1, deg=None, sigma=5.0, offset=0.0):

        self.kernel = kernel
        self.gamma = gamma
        self.deg = deg
        self.sigma = sigma
        self.offset = offset
        self.kfunction = self.kernel_function(kernel)

    def kernel_function(self, kernel):

        if kernel == "linear":

            def f(x, y):
                return np.inner(x, y)

            return f

        # Radial Basis Function
        elif kernel == "rbf":

            def f(x, y):
                return np.exp(-self.gamma * (np.linalg.norm(x - y) ** 2))

            return f

        elif kernel == "polynomial":

            def f(x, y):
                return (self.gamma * (self.offset + np.dot(x, y))) ** self.deg

            return f

        elif kernel == "gaussian":

            def f(x, y):
                return np.exp(-linalg.norm(x - y) ** 2 / (2 * (self.sigma ** 2)))

            return f
        elif kernel == "spectrum":

            def f(x, y):
                inner = 0
                for id_ in x:
                    if id_ in y:
                        inner += x[id_] * y[id_]
                return inner

            return f

    def compute_gram_matrix(self, X):
        n = len(X)
        K = np.zeros((n, n))
        for i in tqdm(range(n)):
            for j in range(i + 1):
                prod_scal = self.kfunction(X[i], X[j])
                K[i, j] = prod_scal
                K[j, i] = prod_scal
        return K


def get_gram_cross(X_train, X_val, kernel="spectrum"):
    """
    get pairwise kernel evaluations between train and val/test.
    """
    ker = Kernel(kernel=kernel)
    n, m = len(X_train), len(X_val)
    gram_cross = np.zeros((n, m))
    for i in tqdm(range(n)):
        for j in range(m):
            gram_cross[i, j] = ker.kfunction(X_train[i], X_val[j])

    return gram_cross