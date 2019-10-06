from deep_fuzzy_k_means import *


class AutoEncoder:
    def __init__(self, data_set, labels, cluster_count, layers, max_iter=-1, delta=1.0, lam2=0.2,
                 file_name='AE', activation=0, read_params=False, rate=0.1, decay_threshold=800):
        self.iteration = 0
        self.X = data_set  # n * d
        self.y = labels
        self.c = cluster_count
        self.delta = delta
        self.lam2 = lam2
        self.rate = rate
        self.batch = np.shape(data_set)[0]
        self.decay_threshold = decay_threshold
        if not file_name == 'AE':
            self.file_name = 'AE_' + file_name
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
                self.b.append(np.zeros((layers[i], 1)))
            self.h.append(np.empty((layers[i], 1)))
            self.a.append(np.empty((layers[i], 1)))
        self.H = np.empty((self.size, self.dim))
        self.HM = np.empty(self.X.shape)
        if read_params:
            self.__read_params()

    def fp(self):
        self.__whole_fp()

    def __cal_J1(self):
        result = np.linalg.norm(self.HM - self.X) ** 2
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
        for W in self.W:
            result += np.linalg.norm(W) ** 2 * self.lam2
        for b in self.b:
            result += np.linalg.norm(b) ** 2 * self.lam2
        return result

    def adaptive_function(self, g):
        norm = np.linalg.norm(g)
        return (1 + self.delta) * norm ** 2 / (norm + self.delta)

    def train(self):
        temp = 0
        self.__whole_fp()
        while not (self.__max_iteration()):
            self.iteration += 1
            if self.iteration >= self.decay_threshold and self.iteration % self.decay_threshold == 0:
                self.rate /= 10
            for i in range(self.batch):
                index = self.__select_a_sample()
                self.__fp_iter(np.matrix(self.X[index]).T.getA())
                self.__back_propagation(self.rate)
            last = temp
            # print('iteration', self.iteration)
            continue
            self.__whole_fp()
            temp = self.__cal_J()
            diff = last - temp
            print('iteration', self.iteration,
                  'diff:', str(diff),
                  'J:', temp,
                  'J3', self.__cal_J3())

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
        return np.ones(x.shape)

    # d & u must be array
    def __back_propagation(self, rate=0.1, return_gradient=False):
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
                res[i] *= self.__activation_derivative(self.a[i])
            gradient_W = np.matmul(res[i], self.h[i - 1].T) + self.lam2 * self.W[i]
            gradient_b = res[i] + self.lam2 * self.b[i]
            gradient_Ws[i] = gradient_W
            gradient_bs[i] = gradient_b
        if return_gradient:
            return gradient_Ws, gradient_bs
        for i in range(1, size):
            self.W[i] -= rate * gradient_Ws[i]
            self.b[i] -= rate * gradient_bs[i]

    def write_params(self):
        f = open(self.file_name, 'wb')
        params = [self.W, self.b, self.H]
        pickle.dump(params, f)
        f.close()

    def __read_params(self):
        if not os.path.exists(self.file_name):
            return
        f = open(self.file_name, 'rb')
        params = pickle.load(f)
        f.close()
        self.W = params[0]
        self.b = params[1]
        self.H = params[2]
        print('Using last parameters...')
