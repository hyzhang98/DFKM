import numpy as np
import struct
import pickle
from sklearn.decomposition import PCA
from scipy import stats
import scipy.io as scio
from sklearn import metrics


class FuzzyKMeans:
    def __init__(self, data_set, cluster_count, gamma=1.0, file_name='FKM', convergence_acc=1.0, print_msg=True):
        self.X = data_set  # type\: array, row of X is a sample, n * d
        self.gamma = gamma
        self.c = cluster_count
        self.sample_dim = self.X.shape[1]
        self.size = self.X.shape[0]
        self.file_name = 'params/' + file_name
        self.convergence_acc = convergence_acc
        self.print_msg = print_msg
        # self.U = np.matmul(np.ones((self.size, 1)), np.ones((1, self.c))) / self.c  # n * c
        self.U = np.random.random((self.size, self.c))
        for i in range(self.size):
            self.U[i] /= np.sum(self.U[i])
        self.C = np.random.random((self.sample_dim, self.c))  # d * c
        self.D = np.random.random((self.size, self.c))  # n * c

    def cal_J(self):
        result = 0
        for i in range(self.size):
            for j in range(self.c):
                u = self.U[i][j]
                if u == 0:
                    continue
                result += u * self.D[i][j] + self.gamma * u * np.log(u)
        return result

    def train(self):
        residual = None
        iteration = 0
        while not self.__convergence(residual) and iteration < 100:
            iteration += 1
            print(iteration)
            if self.print_msg:
                print(self.cal_J(), end=' ')
                print('Iteration ' + str(iteration) + ' with residual: ' + str(residual))
            residual = self.__update_C()
            self.__update_D()
            residual = self.__update_U()
        self.__write_params()

    def __convergence(self, res):
        return res is not None \
               and res <= self.convergence_acc

    def __write_params(self):
        f = open(self.file_name + '.params', 'wb')
        params = [self.U, self.C]
        pickle.dump(params, f)
        f.close()

    def __update_D(self):
        T = np.tile(self.X.T, (1, self.c)) - np.repeat(self.C, self.size, axis=1)
        T = np.sum(np.power(T, 2), axis=0)
        self.D = T.reshape((self.c, self.size)).T
        # for i in range(self.size):
        #     for j in range(self.c):
        #         c = self.C[:, j]  # centroid of cluster j
        #         x = self.X[i]  # sample xi, row vector
        #         self.D[i][j] = np.linalg.norm(c - x) ** 2

    def __update_U(self):
        old = self.U.copy()
        T = np.divide(self.D, self.gamma)
        m = np.min(T, axis=1).reshape((self.size, 1))
        T = np.subtract(T, m)
        self.U = np.exp(-1 * T)
        self.U = np.divide(self.U, np.sum(self.U, axis=1).reshape((self.size, 1)))
        # for i in range(self.size):
        #     d = self.D[i] / self.gamma
        #     m = np.min(d)
        #     if m > 100:
        #         diff = m - 100
        #         d -= diff * np.ones(d.shape)
        #     u = np.exp(-1 * d)
        #     self.U[i] = u / np.sum(u)
        return np.linalg.norm(self.U - old)

    def __update_C(self):
        old = self.C.copy()
        T = self.D * self.U
        C = np.matmul(self.X.T, T)
        A = np.array(np.sum(T, axis=0)).reshape((1, self.c))
        # A = np.matmul(np.ones((self.dim, 1)), A)
        A = np.matmul(np.ones((self.sample_dim, 1)), A) + 10 ** -100
        self.C = C / A
        # for j in range(self.c):
        #     u = np.matrix(self.U[:, j]).T
        #     numerator = np.matmul(self.X.T, u)
            # print(numerator)
            # denominator = np.sum(u) + 10**-50
            # print(self.__U[0])
            # print(numerator.shape)
            # print(denominator)
            # self.C[:, j] = (numerator / denominator).getA()[:, 0]
        return np.linalg.norm(self.C - old)

    def read_params(self, file_name):
        f = open(file_name, 'rb')
        params = pickle.load(f)
        self.U = params[0]
        self.C = params[1]
        f.close()

    def print_max_prob(self, n):
        for i in range(n):
            index = np.random.randint(0, self.size)
            print(np.max(self.U[index]))


def cal_acc2(predict_y, y, c):
    size = len(y)
    G = []
    for i in range(c):
        G.append([])
    for i in range(size):
        G[predict_y[i]].append(y[i])
    count = 0
    for g in G:
        if len(g) is 0:
            continue
        count += stats.mode(g).count[0]
    return count / size


def cal_acc(U, y):
    size = U.shape[0]
    predict_y = []
    for i in range(size):
        predict_y.append(np.argmax(U[i]))
    return metrics.accuracy_score(y, predict_y)


def cal_purity(U, y):
    size = U.shape[0]
    c = U.shape[1]
    G = []
    for i in range(c):
        G.append([])
    for i in range(size):
        u = U[i]
        index = np.argmax(u)
        G[index].append(y[i])
    count = 0
    for g in G:
        if len(g) is 0:
            print('---')
            continue
        count += stats.mode(g).count[0]
    return count / size


def reduce_dimensions(X):
    pca = PCA(n_components=300)
    new_X = pca.fit_transform(X)
    return new_X


def get_images():
    # file_name = 'COIL20_HOG_PCA300.mat'
    # file_name = '/Users/hyzhang/Downloads/2018-TIP-StructAE/data/YaleB_LPQ_PCA300_TValid.mat'
    # file_name = 'COIL20_DSIFT_SR_PCA300_TValid.mat'
    file_name = 'data/YaleB_DSIFT_SR_PCA300_TValid.mat'
    # file_name = 'mini_mnist_PCA300.mat'
    mat = scio.loadmat(file_name)
    # X = mat['Data']
    # X = mat['X']
    X = mat['tt_dat'].T
    # X = X / np.max(X)
    # y = mat['tt_labels'][0]
    y = mat['tt_labels'][0]
    # y = mat['Label'][:, 0]
    # y = mat['Y'][:, 0]
    # y = mat['y']
    return X, y


def save_mat(U, labels):
    n, c = U.shape
    size = labels.shape[0]
    predict_y = []
    for i in range(n):
        predict_y.append(np.argmax(U[i]))
    predict_y = np.matrix(predict_y).reshape((size, 1)) + 1
    y = labels.reshape((size, 1))
    mat_name = 'fkm.mat'
    scio.savemat(mat_name, {'y_predicted': predict_y, 'y': y})


if __name__ == '__main__':
    X, y = get_images()
    # f = open('FKM.params', 'rb')
    # params = pickle.load(f)
    # U = params[0]
    # save_mat(U, y)
    # f.close()
    # print(X.shape)
    # print(y)
    print(X.shape)
    fkm = FuzzyKMeans(X, 38, convergence_acc=0.001, gamma=0.1)
    # fkm.read_params('FKM.params')
    fkm.train()
    # save_mat(fkm.U, y)
    print(cal_purity(fkm.U, y))
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=38, random_state=666)
    y_pre = km.fit_predict(X)
    print(cal_acc2(y_pre, y, 38))
    # print(y_pre)
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
    print(np.max(fkm.U[np.random.randint(0, X.shape[0])]))
