import torch
import numpy as np
import utils
from metrics import cal_clustering_metric
from scipy.sparse import coo_matrix
from sklearn.cluster import KMeans
import scipy.io as scio
import random
import warnings

warnings.filterwarnings('ignore')


class Dataset(torch.utils.data.Dataset):
    def __init__(self, X):
        self.X = X

    def __getitem__(self, idx):
        return self.X[:, idx], idx

    def __len__(self):
        return self.X.shape[1]


class PretrainDoubleLayer(torch.nn.Module):
    def __init__(self, X, dim, device, act, batch_size=128, lr=10**-3):
        super(PretrainDoubleLayer, self).__init__()
        self.X = X
        self.dim = dim
        self.lr = lr
        self.device = device
        self.enc = torch.nn.Linear(X.shape[0], self.dim)
        self.dec = torch.nn.Linear(self.dim, X.shape[0])
        self.batch_size = batch_size
        self.act = act

    def forward(self, x):
        if self.act is not None:
            z = self.act(self.enc(x))
            return z, self.act(self.dec(z))
        else:
            z = self.enc(x)
            return z, self.dec(z)

    def _build_loss(self, x, recons_x):
        size = x.shape[0]
        return torch.norm(x-recons_x, p='fro')**2 / size

    def run(self):
        self.to(self.device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        train_loader = torch.utils.data.DataLoader(Dataset(self.X), batch_size=self.batch_size, shuffle=True)
        loss = 0
        for epoch in range(10):
            for i, batch in enumerate(train_loader):
                x, _ = batch
                optimizer.zero_grad()
                _, recons_x = self(x)
                loss = self._build_loss(x, recons_x)
                loss.backward()
                optimizer.step()
            print('epoch-{}: loss={}'.format(epoch, loss.item()))
        Z, _ = self(self.X.t())
        return Z.t()


class DeepFuzzyKMeans(torch.nn.Module):
    def __init__(self, X, labels, layers=None, lam=1, sigma=None, gamma=1, lr=10**-3, device=None, batch_size=128):
        super(DeepFuzzyKMeans, self).__init__()
        if layers is None:
            layers = [X.shape[0], 500, 300]
        if device is None:
            device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
        self.layers = layers
        self.device = device
        if not isinstance(X, torch.Tensor):
            X = torch.Tensor(X)
        self.X = X.to(device)
        self.labels = labels
        self.gamma = gamma
        self.lam = lam
        self.sigma = sigma
        self.batch_size = batch_size
        self.n_clusters = len(np.unique(self.labels))
        self.lr = lr
        self._build_up()

    def _build_up(self):
        self.act = torch.tanh
        self.enc1 = torch.nn.Linear(self.layers[0], self.layers[1])
        self.enc2 = torch.nn.Linear(self.layers[1], self.layers[2])
        self.dec1 = torch.nn.Linear(self.layers[2], self.layers[1])
        self.dec2 = torch.nn.Linear(self.layers[1], self.layers[0])

    def forward(self, x):
        z = self.act(self.enc1(x))
        z = self.act(self.enc2(z))
        recons_x = self.act(self.dec1(z))
        recons_x = self.act(self.dec2(recons_x))
        return z, recons_x

    def _build_loss(self, z, x, d, u, recons_x):
        size = x.shape[0]
        loss = 1/2 * torch.norm(x - recons_x, p='fro') ** 2 / size
        t = d*u  # t: m * c
        distances = utils.distance(z.t(), self.centroids)
        loss += (self.lam / 2 * torch.trace(distances.t().matmul(t)) / size)
        loss += 10**-5 * (self.enc1.weight.norm()**2 + self.enc1.bias.norm()**2) / size
        loss += 10**-5 * (self.enc2.weight.norm()**2 + self.enc2.bias.norm()**2) / size
        loss += 10**-5 * (self.dec1.weight.norm()**2 + self.dec1.bias.norm()**2) / size
        loss += 10**-5 * (self.dec2.weight.norm()**2 + self.dec2.bias.norm()**2) / size
        return loss

    def run(self):
        self.to(self.device)
        self.pretrain()
        Z, _ = self(self.X.t())
        Z = Z.t().detach()
        idx = random.sample(list(range(Z.shape[1])), self.n_clusters)
        self.centroids = Z[:, idx] + 10 ** -6
        self._update_U(Z)
        # D = self._update_D(Z)
        # self.clustering(D, Z)
        print('Starting training......')
        train_loader = torch.utils.data.DataLoader(Dataset(self.X), batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        loss = 0
        for epoch in range(30):
            D = self._update_D(Z)
            for i, batch in enumerate(train_loader):
                x, idx = batch
                optimizer.zero_grad()
                z, recons_x = self(x)
                d = D[idx, :]
                u = self.U[idx, :]
                loss = self._build_loss(z, x, d, u, recons_x)
                loss.backward()
                optimizer.step()
            Z, _ = self(self.X.t())
            Z = Z.t().detach()
            # D = self._update_D(Z)
            # for i in range(20):
            self.clustering(Z, 1)
            _, y_pred = self.U.max(dim=1)
            y_pred = y_pred.detach().cpu() + 1
            y_pred = y_pred.numpy()
            acc, nmi = cal_clustering_metric(self.labels, y_pred)
            print('epoch-{}, loss={}, ACC={}, NMI={}'.format(epoch, loss.item(), acc, nmi))

    def pretrain(self):
        string_template = 'Start pretraining-{}......'
        print(string_template.format(1))
        pre1 = PretrainDoubleLayer(self.X, self.layers[1], self.device, self.act, lr=self.lr)
        Z = pre1.run()
        self.enc1.weight = pre1.enc.weight
        self.enc1.bias = pre1.enc.bias
        self.dec2.weight = pre1.dec.weight
        self.dec2.bias = pre1.dec.bias
        print(string_template.format(2))
        pre2 = PretrainDoubleLayer(Z.detach(), self.layers[2], self.device, self.act, lr=self.lr)
        pre2.run()
        self.enc2.weight = pre2.enc.weight
        self.enc2.bias = pre2.enc.bias
        self.dec1.weight = pre2.dec.weight
        self.dec1.bias = pre2.dec.bias

    def _update_D(self, Z):
        if self.sigma is None:
            return torch.ones([Z.shape[1], self.centroids.shape[1]]).to(self.device)
        else:
            distances = utils.distance(Z, self.centroids, False)
            return (1 + self.sigma) * (distances + 2 * self.sigma) / (2 * (distances + self.sigma))

    def clustering(self, Z, max_iter=1):
        for i in range(max_iter):
            D = self._update_D(Z)
            T = D * self.U
            self.centroids = Z.matmul(T) / T.sum(dim=0).reshape([1, -1])
            self._update_U(Z)

    def _update_U(self, Z):
        if self.sigma is None:
            distances = utils.distance(Z, self.centroids, False)
        else:
            distances = adaptive_loss(utils.distance(Z, self.centroids, False), self.sigma)
        U = torch.exp(-distances / self.gamma)
        self.U = U / U.sum(dim=1).reshape([-1, 1])


def adaptive_loss(D, sigma):
    return (1 + sigma) * D * D / (D + sigma)


if __name__ == '__main__':
    import data_loader as loader
    data, labels = loader.load_data(loader.USPS)
    data = data.T
    for lam in [10**-3, 10**-2, 10**-1, 1]:
        print('lam={}'.format(lam))
        dfkm = DeepFuzzyKMeans(data, labels, [data.shape[0], 512, 300], lam=lam, gamma=1, batch_size=512, lr=10**-4)
        dfkm.run()
