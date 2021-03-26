import cvxopt
import cvxopt.solvers
import numpy as np


class SVM:
    # init function for the SVM classifier
    def __init__(self, gram_m, gamma=0.1, deg=3, C=1.0, offset=0, sigma=5.0):
        self.offset = offset
        self.sigma = sigma
        self.C = C
        self.K = gram_m
        if self.C is not None:
            self.C = float(self.C)

    def fit(self, y):

        n_s = len(self.K)

        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(np.ones(n_s) * -1)
        A = cvxopt.matrix(y, (1, n_s), "d")
        b = cvxopt.matrix(0.0)

        if self.C is None:
            G = cvxopt.matrix(np.diag(np.ones(n_s) * -1))
            h = cvxopt.matrix(np.zeros(n_s))
        else:
            G = cvxopt.matrix(
                np.vstack((np.diag(np.ones(n_s) * -1), np.identity(n_s)))
            )
            h = cvxopt.matrix(np.hstack((np.zeros(n_s), np.ones(n_s) * self.C)))

        # Obtaining Lagrange multipliers
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        a = np.ravel(solution["x"])

        # Support vectors have "non zero" lagrange multipliers -> threshold = 1e-5
        sv = a > 1e-5
        ind = np.arange(len(a))[sv]
        self.sv = sv
        self.a = a[sv]
        self.sv_y = y[sv]

        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * self.K[ind[n], sv])
        self.b /= len(self.a)

    def predict(self, gram_val):
        """ predict for a set of examples described by gram_val """
        m = gram_val.shape[1]
        y_predict = np.array(
            [
                (self.a * self.sv_y * gram_val[self.sv, i]).sum()
                for i in range(m)
            ]
        )
        return y_predict + self.b


class KRL:
    """
    Class for kernel logistic regression.
    """

    def __init__(self, gram_m, gamma=0.01, max_iter=100, lambd=0.1):

        self.K = gram_m
        self.lambda_ = lambd
        self.max_iter_ = max_iter

    def fit(self, y):
        """
        Fit the data (x, y).

        """

        def compute_loss(y_):
            """
            Compute the logistic loss over all observations.
            """

            L = np.mean(
                [
                    np.log(1 + np.exp(-y_[i] * (self.K @ self.alpha_)[i]))
                    for i in range(len(self.K))
                ]
            )
            return L

        n = len(self.K)
        # We initialize coefficients from the normal distribution , we take a standard deviation relatively small
        self.alpha_ = np.random.normal(0, 0.01, n)
        n_iter = 0

        old_loss, new_loss = 0, np.inf
        while n_iter < self.max_iter_:

            z, W = self.update_zi(np.array(y), self.K @ self.alpha_)
            W = np.diag(W)
            self.alpha_ = self.update_coef(z, W, self.K)
            # print('loss for iter {} is {}'.format(n_iter,compute_loss(y)))

            n_iter += 1

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def update_coef(self, z, W, K):
        """
        Update alpha coefficients by solving weighted kernel ridge regression problem.
        """
        W_sqrt = np.sqrt(W)
        return (
            W_sqrt
            @ np.linalg.inv(
                W_sqrt @ K @ W_sqrt
                + K.shape[0] * self.lambda_ * np.eye(K.shape[0])
            )
            @ W_sqrt
            @ z
        )

    def update_zi(self, y, m):
        """
        Get  zi as defined in slide  114.
        """
        Pi = -self.sigmoid(-y * m)
        Wi = self.sigmoid(m) * self.sigmoid(-m)
        zi = m - Pi * y / Wi
        return zi, Wi

    def predict_single_point(self, gram_val, i):
        """
        Predict the label of one vector.
        """
        """
        pred = np.array([self.alpha_[i] * self.kernel_instance._function(self.X_train[i],x) 
                         for i in range(len(self.X_train)) ])
        """
        return (self.alpha_ * gram_val[:, i]).sum()

    def predict(self, gram_val):
        """
        predict labels for a matrix X(n,p).
        """
        return np.array(
            [
                self.predict_single_point(gram_val, i)
                for i in range(gram_val.shape[1])
            ]
        )
