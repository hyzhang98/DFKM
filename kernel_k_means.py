import numpy as np


class KernelKMeans:
    GAUSSIAN = 0

    def __init__(self, X, k, gamma=100.0, kernel=GAUSSIAN):
        self.gamma = gamma
        self.X = X
        self.k = k
        self.size = X.shape[0]
        self.ker = kernel
        self.clusters = []
        for i in range(self.size):
            self.clusters.append([])
        self.energies = np.zeros(self.k).tolist()

    def train(self):
        assignment = np.random.randint(0, self.k, self.size).tolist()
        assert isinstance(assignment, list)
        # print(self.clusters)
        max_iteration = 100
        iteration = 0
        while iteration < max_iteration:
            iteration += 1
            print('iteration', iteration)
            self.to_cluster(assignment)
            old = assignment[:]
            self.__cal_energies()
            for i in range(self.size):
                min_dist = 2
                min_k = 0
                for k in range(self.k):
                    d = self.__cal_dist(self.X[i], k)
                    if d < min_dist:
                        min_k = k
                        min_dist = d
                assignment[i] = min_k
            if assignment == old:
                print(iteration)
                break
        return np.array(assignment)

    def __cal_dist(self, x, k):
        value = self.kernel(x, x)
        n = len(self.clusters[k])
        if n == 0:
            return 10**10
        temp = 0
        for i_y in self.clusters[k]:
            y = self.X[i_y]
            temp += self.kernel(x, y)
        value -= 2 * temp / n
        value += self.energies[k]
        return value

    def __cal_energies(self):
        assert isinstance(self.energies, list)
        for i in range(self.k):
            self.energies[i] = self.__cal_energy(i)

    def __cal_energy(self, k):
        value = 0
        n = len(self.clusters[k])
        if n == 0:
            return 10**10
        for i_b in self.clusters[k]:
            for i_c in self.clusters[k]:
                b = self.X[i_b]
                c = self.X[i_c]
                value += self.kernel(b, c)
        value /= n**2
        return value

    def to_cluster(self, assignment):
        self.clusters = []
        for i in range(self.size):
            self.clusters.append([])
        for i in range(self.size):
            self.clusters[assignment[i]].append(i)

    def kernel(self, x, y):
        return np.exp(-np.linalg.norm(x-y)**2/self.gamma)


if __name__ == '__main__':
    from deep_fuzzy_k_means import *
    X, y = get_images(YALE)
    X = X.astype(np.float64)
    X /= np.max(X)
    cluster_count = 15
    kkm = KernelKMeans(X, cluster_count, 100)
    y_predicted = kkm.train()
    print(cal_metric2(y_predicted, y, cluster_count))
