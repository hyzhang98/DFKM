import pickle
import struct
from sklearn.decomposition import PCA
import os
import random
from fuzzy_k_means import FuzzyKMeans
from utils import *
from matplotlib import pyplot as plt


TRICK = False


TANH = 0
SIGMOID = 1
RELU = 2


COIL20_SDSIFT = 0
YALE_B_SDSIFT = 1
COIL20_RAW = 2
COIL20_HOG = 3
MNIST_MINI = 4
WINE = 5
YALE = 6
ORL = 7
BINALPHA = 8
UMIST = 9
JAFFE = 10
PALM = 11
USPS = 12  # 20% samples
YALE_B_LPQ = 13
CIFAR10 = 14
FILE_NAMES = ['COIL20_DSIFT_SR_PCA300_TValid.mat',
              'YaleB_DSIFT_SR_PCA300_TValid.mat',
              'Coil20Data_25_uni.mat',
              'COIL20_HOG_PCA300.mat',
              'mini_mnist_PCA300.mat',
              'wine_uni.mat',
              'Yale.mat',
              'ORL.mat',
              'binalpha_uni.mat',
              'umist_1024.mat',
              'jaffe.mat',
              'PalmData25_uni.mat',
              'USPSdata_20_uni.mat',
              'YaleB_LPQ_PCA300_TValid.mat',
              'cifar-10-pca300.mat']
PARAMS_NAMES = ['COIL20_SDSIFT', 'YaleB', 'COIL20_RAW', 'COIL_HOG', 'mini_mnist_PCA300', 'wine', 'Yale', 'ORL',
                'BinAlpha', 'UMIST', 'JAFFE', 'Palm', 'USPS_20', 'YaleB_LPQ', 'CIFAR_10']


class DeepFuzzyKMeans:
    def __init__(self, data_set, labels, cluster_count, layers, max_iter=-1, gamma=1.0, delta=1.0, lam1=1.0, lam2=0.2,
                 file_name='DFKM', activation=0, read_params=True, rate=0.1, batch=100,
                 decay_threshold=800, fuzzy_max_iter=15):
        '''
        :param data_set: Training set, n * d
        :param labels: Training labels, n * 1
        :param cluster_count:
        :param layers: the number of each layers, for example, the corresponding input of a five
                        auto-encoder is [300, 200, 100, 200, 300];
        :param max_iter: Max Iteration
        :param gamma: The trade-off coefficient of entropy term
        :param delta: Hyper-parameter of the adaptive loss
        :param lam1: The trade-off coefficient of reconstruction errors
        :param lam2: The trade-off coefficient of Ridge regression
        :param file_name: Name of the saved file
        :param activation: The type of activation function: 0-tanh, 1-sigmoid, 2-ReLU.
        :param read_params: If set it as True, it will use the old parameters of auto-encoder
        :param rate: learning rate
        :param batch: The number of samples used in each iteration
        :param decay_threshold: The learning rate will decay per <decay_threshold> iteration.
        :param fuzzy_max_iter: When perform multiple steps to update fuzzy model, it is the maximum iterations.
                                    The switch is the TRICK variable.
        '''
        self.iteration = 0
        self.X = data_set  # n * d
        self.y = labels
        self.c = cluster_count
        self.gamma = gamma
        self.delta = delta
        self.lam1 = lam1
        self.lam2 = lam2
        self.rate = rate
        self.batch = batch
        self.decay_threshold = decay_threshold
        self.fuzzy_max_iter = fuzzy_max_iter
        if not file_name == 'DFKM':
            self.file_name = 'DFKM_' + file_name
        else:
            self.file_name = file_name
        self.file_name = 'params/' + self.file_name + '.params'
        self.activation = activation
        self.size = self.X.shape[0]
        layer_count = len(layers)
        self.dim = layers[int((layer_count - 1) / 2)]
        if max_iter is -1:
            self.max_iter = 100
        else:
            self.max_iter = max_iter
        # self.U = np.matmul(np.ones((self.size, 1)), np.ones((1, self.c))) / self.c
        self.U = np.random.random((self.size, self.c))
        for i in range(self.size):
            self.U[i] /= np.sum(self.U[i])
        self.L = np.empty((self.size, self.c))  # ||xi - cj||
        self.D = np.empty((self.size, self.c))
        self.C = np.eye(self.dim, self.c)  # k * c
        # self.C = (np.random.random((self.dim, self.c)) - 0.5) * 10  # k * c
        self.W = []
        self.b = []
        self.h = []
        self.a = []
        for i in range(layer_count):
            if i == 0:
                self.W.append(0)
                self.b.append(0)
            else:
                self.W.append(np.eye(layers[i], layers[i - 1]))
                # self.W.append(np.random.random((layers[i], layers[i - 1])) - 0.5)
                # self.b.append(np.zeros((layers[i], 1)) + 0.1)
                self.b.append(np.zeros((layers[i], 1)))
            self.h.append(np.empty((layers[i], 1)))
            self.a.append(np.empty((layers[i], 1)))
        self.H = np.empty((self.size, self.dim))
        self.HM = np.empty(self.X.shape)
        self.Us = []
        self.purs = []
        self.nmis = []
        if read_params:
            self.__read_params()

    def fp(self):
        self.__whole_fp()

    def __cal_J1(self):
        result = np.linalg.norm(self.HM - self.X) ** 2
        return result

    def __cal_J2(self):
        result = 0
        for i in range(self.size):
            for j in range(self.c):
                u = self.U[i][j]
                t = u * self.adaptive_function(self.H[i] - self.C[:, j])
                if not u == 0:
                    t += + self.gamma * u * np.log(u)
                result += self.lam1 * t
        return result

    def __cal_J3(self):
        result = 0
        for W in self.W:
            result += np.linalg.norm(W) ** 2 * self.lam2
        for b in self.b:
            result += np.linalg.norm(b) ** 2 * self.lam2
        return result

    def __cal_J(self):
        result = np.linalg.norm(self.HM - self.X) ** 2
        for i in range(self.size):
            for j in range(self.c):
                u = self.U[i][j]
                t = u * self.adaptive_function(self.H[i] - self.C[:, j])
                if not u == 0:
                    t += self.gamma * u * np.log(u)
                result += self.lam1 * t
        for W in self.W:
            result += np.linalg.norm(W) ** 2 * self.lam2
        for b in self.b:
            result += np.linalg.norm(b) ** 2 * self.lam2
        return result

    def adaptive_function(self, g):
        norm = np.linalg.norm(g)
        return (1 + self.delta) * norm ** 2 / (norm + self.delta)

    def train(self):
        diff = 10 ** 4
        temp = 0
        while not (self.__max_iteration()):
            self.iteration += 1
            if self.iteration >= self.decay_threshold and self.iteration % self.decay_threshold == 0:
                self.rate /= 10
            self.__whole_fp()
            self.__update_L()
            self.__update_D()
            # for i in range(self.size):
            for i in range(self.batch):
                index = self.__select_a_sample()
                self.__fp_iter(np.matrix(self.X[index]).T.getA())
                d = np.matrix(self.D[index]).T.getA()
                u = np.matrix(self.U[index]).T.getA()
                self.__back_propagation(d, u, self.rate)
            delta = 100
            k = 0
            if TRICK:
                while delta > 0.01 and k < self.fuzzy_max_iter:
                    self.__update_C()
                    self.__update_L()
                    delta = self.__update_U()
                    k += 1
            else:
                self.__update_C()
                self.__update_L()
                self.__update_D()
                delta = self.__update_U()
            self.Us.append(self.U.copy())
            # if not self.iteration % 10 == 0:
            if not self.iteration % 20 == 0:
                continue
            last = temp
            self.__whole_fp()
            temp = self.__cal_J()
            diff = last - temp
            t2 = self.__cal_J2()
            if self.y is None:
                print('iteration', self.iteration,
                      'diff:', str(diff),
                      'J:', temp,
                      'J2:', t2,
                      'J3', self.__cal_J3())
                continue
            acc = cal_acc(self.U, self.y)
            mi = cal_nmi(self.U, self.y)
            self.purs.append(acc)
            self.nmis.append(mi)
            print('iteration', self.iteration,
                  'diff:', str(diff),
                  'J:', temp,
                  'J2:', t2,
                  'J3', self.__cal_J3(),
                  'acc', acc,
                  'nmi:', mi)
            if self.iteration % 10 == 0:
                print('Iteration ' + str(self.iteration) + ' and purity: '
                      + str(acc) + ' delta: ' + str(delta))
                print(np.max(self.U[self.__select_a_sample()]))
                print(np.max(self.U[self.__select_a_sample()]))
                print(np.max(self.U[self.__select_a_sample()]))


    def __max_iteration(self):
        return self.iteration >= self.max_iter

    def __select_a_sample(self):
        return np.random.randint(0, self.size)

    def __whole_fp(self):
        size = len(self.h)
        H = self.X
        for i in range(1, size):
            z = np.matmul(self.W[i], H.T) + np.matmul(self.b[i], np.ones((1, self.size)))
            H = self.__activation(z).T
            if i == (size - 1) / 2:
                self.H = H
            elif i == size - 1:
                self.HM = H

    def __fp_iter(self, h0):
        x = h0
        size = len(self.h)
        self.h[0] = x
        for i in range(1, size):
            a = np.matmul(self.W[i], x) + self.b[i]
            x = self.__activation(a)
            self.h[i] = x
            self.a[i] = a

    def __activation(self, x):
        if self.activation == 0:
            return np.tanh(x)
        elif self.activation == SIGMOID:
            return sigmoid(x)
        return x

    # return array
    def __activation_derivative(self, x):
        if self.activation == 0:
            return np.ones(x.shape) - np.tanh(x) ** 2
        elif self.activation == SIGMOID:
            return sigmoid(x) * (1 - sigmoid(x))
        elif self.activation == RELU:
            return np.where(x <= 0, 0, 1)
        return np.ones(x.shape)

    # d & u must be array
    def __back_propagation(self, d, u, rate=0.1, return_gradient=False):
        size = len(self.h)
        res = []
        gradient_Ws = []
        gradient_bs = []
        for i in range(size):
            res.append(0)
            gradient_Ws.append(0)
            gradient_bs.append(0)
        for i in reversed(range(1, size)):
            if i == size - 1:
                res[i] = (self.h[size - 1] - self.h[0]) * self.__activation_derivative(self.a[size - 1])
            else:
                res[i] = np.matmul(self.W[i + 1].T, res[i + 1])
                if i == (size - 1) / 2:
                    h = np.matrix(self.h[i]).getA()
                    P = np.matmul(h, np.ones((1, self.c))) - self.C
                    alpha = d * u
                    # print('P: ' + str(P.shape) + ' alpha: ' + str(alpha.shape))
                    lam = np.matmul(P, alpha)
                    res[i] += self.lam1 * lam
                res[i] *= self.__activation_derivative(self.a[i])
            gradient_W = np.matmul(res[i], self.h[i - 1].T) + self.lam2 * self.W[i]
            gradient_b = res[i] + self.lam2 * self.b[i]
            gradient_Ws[i] = gradient_W
            gradient_bs[i] = gradient_b
            # gradient_Ws.append([i, gradient_W])
            # gradient_bs.append([i, gradient_b])
        if return_gradient:
            return gradient_Ws, gradient_bs
        for i in range(1, size):
            self.W[i] -= rate * gradient_Ws[i]
            self.b[i] -= rate * gradient_bs[i]

    def __update_L(self):
        T = np.tile(self.H.T, (1, self.c)) - np.repeat(self.C, self.size, axis=1)
        T = np.sum(np.power(T, 2), axis=0)
        self.L = T.reshape((self.c, self.size)).T
        # for j in range(self.c):
        #     T = self.H.T - np.matmul(np.array(self.C[:, j]).reshape((self.dim, 1)), np.ones((1, self.size)))
        #     self.L[:, j] = np.linalg.norm(T, axis=0)**2

    def __update_D(self):
        Lt = np.sqrt(self.L)
        self.D = (Lt + 2 * self.delta) * (1 + self.delta)
        T = 2 * np.power(Lt + self.delta, 2)
        self.D = np.divide(self.D, T)
        # for i in range(self.size):
        #     h = self.H[i]
        #     for j in range(self.c):
        #         norm = np.linalg.norm(h - self.C[:, j])
        #         self.D[i][j] = (1 + self.delta) * (norm + 2 * self.delta) / (2 * (norm + self.delta) ** 2)

    def __update_U(self):
        old = self.U.copy()
        if TRICK:
            T = self.D * self.L
        else:
            T = np.sqrt(self.L) + self.delta
            T = (1 + self.delta) * self.L / T
        # for i in range(self.size):
        #     t = T[i] / self.gamma
        #     m = np.min(t)
        #     if m > 100:
        #         diff = m - 100
        #         t -= diff * np.ones(t.shape)
        #     u = np.exp(-1 * t)
        #     self.U[i] = u / np.sum(u)
        T = np.divide(T, self.gamma)
        m = np.min(T, axis=1).reshape((self.size, 1))
        T = np.subtract(T, m)
        self.U = np.exp(-1 * T)
        self.U = np.divide(self.U, np.sum(self.U, axis=1).reshape((self.size, 1)))
        return np.linalg.norm(self.U - old)

    def __update_C(self):
        old = self.C.copy()
        T = self.D * self.U
        C = np.matmul(self.H.T, T)
        A = np.array(np.sum(T, axis=0)).reshape((1, self.c))
        # A = np.matmul(np.ones((self.dim, 1)), A)
        A = np.matmul(np.ones((self.dim, 1)), A) + 10**-100
        self.C = C / A
        return np.linalg.norm(self.C - old)

    def write_params(self):
        f = open(self.file_name, 'wb')
        params = [self.U, self.C, self.W, self.b]
        pickle.dump(params, f)
        f.close()

    def __read_params(self):
        if not os.path.exists(self.file_name):
            return
        f = open(self.file_name, 'rb')
        params = pickle.load(f)
        f.close()
        self.U = params[0]
        self.C = params[1]
        self.W = params[2]
        self.b = params[3]
        print('Using last parameters...')


def reduce_dimensions(X):
    pca = PCA(n_components=300)
    new_X = pca.fit_transform(X)
    return new_X


def get_mnist():
    image_file = open('mnist-test-10k-images-ubyte', 'rb')
    magic, count, rows, columns = struct.unpack('>IIII', image_file.read(16))
    images = []
    size = rows * columns
    unpack_mode = str(size) + 'B'
    for i in range(count):
        image = image_file.read(size)
        image = struct.unpack(unpack_mode, image)
        # image = np.array(image)
        images.append(image)
    image_file.close()
    X = np.array(images)
    label_file = open('mnist-test-10k-labels-ubyte', 'rb')
    magic, count = struct.unpack('>II', label_file.read(8))
    labels = []
    for i in range(count):
        labels.append(struct.unpack('1B', label_file.read(1))[0])
    label_file.close()
    y = np.array(labels)
    X = reduce_dimensions(X)
    X /= np.max(X)
    mat_name = 'mnist_PCA300.mat'
    scio.savemat(mat_name, {'X': X, 'y': y})
    randlist = random.sample(range(0, count), int(count / 10))
    mini_X = []
    mini_y = []
    for i in randlist:
        mini_X.append(X[i])
        mini_y.append(y[i])
    mini_X = np.array(mini_X)
    mini_y = np.array(mini_y).reshape((int(count/10), 1))
    mat_name = 'mini_mnist_PCA300.mat'
    scio.savemat(mat_name, {'X': mini_X, 'y': mini_y})
    print(mini_X.shape)
    print(mini_y.shape)
    print(X.shape)


def get_images(data_set):
    file_name = 'data/' + FILE_NAMES[data_set]
    mat = scio.loadmat(file_name)
    if data_set == YALE_B_SDSIFT:
        X = mat['tt_dat'].T
        y = mat['tt_labels'][0]  # YaleB
    elif data_set in [COIL20_SDSIFT, COIL20_HOG]:
        X = mat['Data']
        y = mat['Label'][:, 0]  # COIL20-300
    else:
        X = mat['X']  # COIL20
        y = mat['Y'][:, 0]  # COIL20
    return X, y


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.where(x < 0, 0, x)
