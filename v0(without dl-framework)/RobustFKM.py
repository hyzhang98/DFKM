import numpy as np


class RobustFKM:
    def __init__(self, X, y, k, gamma, convergence_acc=0.1, delta=1, print_msg=True):
        self.gamma = gamma
        self.X = X  # n * d
        self.y = y
        self.k = k
        self.delta = delta
        self.print_msg = print_msg
        self.convergence_acc = convergence_acc
        self.size, self.dim = self.X.shape
        self.U = np.random.random((self.size, self.k))
        for i in range(self.size):
            self.U[i] /= np.sum(self.U[i])
        self.L = np.empty((self.size, self.k))  # ||xi - cj||
        self.D = np.empty((self.size, self.k))
        self.C = np.random.random((self.dim, self.k))

    def __update_D(self):
        Lt = np.sqrt(self.L)
        self.D = (Lt + 2 * self.delta) * (1 + self.delta)
        T = 2 * np.power(Lt + self.delta, 2)
        self.D = np.divide(self.D, T)

    def __update_L(self):
        T = np.tile(self.X.T, (1, self.k)) - np.repeat(self.C, self.size, axis=1)
        T = np.sum(np.power(T, 2), axis=0)
        self.L = T.reshape((self.k, self.size)).T

    def __update_U(self):
        old = self.U.copy()
        T = np.sqrt(self.L) + self.delta
        T = (1 + self.delta) * self.L / T
        # T = self.D * self.L
        T = np.divide(T, self.gamma)
        m = np.min(T, axis=1).reshape((self.size, 1))
        T = np.subtract(T, m)
        self.U = np.exp(-1 * T)
        self.U = np.divide(self.U, np.sum(self.U, axis=1).reshape((self.size, 1)))
        return np.linalg.norm(self.U - old)

    def __update_C(self):
        old = self.C.copy()
        T = self.D * self.U
        C = np.matmul(self.X.T, T)
        A = np.array(np.sum(T, axis=0)).reshape((1, self.k))
        A = np.matmul(np.ones((self.dim, 1)), A) + 10**-100
        self.C = C / A
        #return np.linalg.norm(self.C - old)

    def __cal_J(self):
        loss = 0
        for i in range(self.size):
            for j in range(self.k):
                u = self.U[i][j]
                x = self.X[i]
                c = self.C[:, j]
                if u == 0:
                    continue
                loss += u * adaptive_loss_function(x-c, self.delta)
                loss += u * np.log(u)
        return loss

    def train(self):
        res = 100
        iteration = 0
        while not self.converge(res) and iteration < 300:
            iteration += 1
            print('iteration', iteration, 'res', res)
            self.__update_L()
            self.__update_D()
            self.__update_C()
            # print('iterations', iteration, 'loss: ', self.__cal_J(), 'res: ', res)
            self.__update_L()
            self.__update_D()
            res = self.__update_U()
            if self.print_msg:
                print('iterations', iteration, 'loss:', self.__cal_J(), 'res: ', res)

    def converge(self, loss):
        return loss <= self.convergence_acc


def adaptive_loss_function(x, sigma):
    norm = np.linalg.norm(x)
    return (1 + sigma) * norm ** 2 / (norm + sigma)


if __name__ == '__main__':
    from deep_fuzzy_k_means import *
    X, y = get_images(YALE)
    X = X.astype(np.float64)
    X /= 255
    cluster_count = 15
    rfkm = RobustFKM(X, y, cluster_count, 3, 0.01, delta=15)
    rfkm.train()
    print(cal_acc(rfkm.U, y))
