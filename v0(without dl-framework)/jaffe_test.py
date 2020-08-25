from deep_fuzzy_k_means import *
from auto_encoder import AutoEncoder
from utils import *

DIR_NAME = 'results/jaffe/'
KM = 0
FKM = 1
SC = 2
AE = 3


def train_DFKM(X, y, k):
    '''
    :param X: Training Set, n * d
    :param y: Training Labels, n * 1
    :param k: Amount of clusters
    :return:
    '''

    f = open('params/jaffe_fkm.params', 'rb')
    params = pickle.load(f)
    U = params[0]
    print(cal_metric(U, y))
    f.close()
    lam1 = 1.5
    lam2 = 10 ** -3
    gamma = 0.01
    delta = 0.1
    rate = 0.001
    activation = TANH
    print('-----------------start pre-training 1-----------------')
    pre_1 = DeepFuzzyKMeans(X, y, k, [1024, 512, 1024], max_iter=20, gamma=gamma, delta=delta,
                            lam1=lam1, lam2=lam2, file_name=PARAMS_NAMES[JAFFE], read_params=False, rate=rate,
                            activation=activation)
    pre_1.U = U
    pre_1.train()
    print('-----------------start pre-training 2-----------------')
    pre_2 = DeepFuzzyKMeans(pre_1.H, y, k, [512, 300, 512], max_iter=20, gamma=gamma, delta=delta,
                            lam1=lam1, lam2=lam2, file_name=PARAMS_NAMES[JAFFE], read_params=False, rate=rate,
                            activation=activation)
    pre_2.U = pre_1.U
    pre_2.train()
    print('-----------------start training-----------------')
    dfkm = DeepFuzzyKMeans(X, y, k, [1024, 512, 300, 512, 1024], max_iter=100, gamma=gamma, delta=delta,
                           lam1=lam1, lam2=lam2, file_name=PARAMS_NAMES[JAFFE], read_params=False, rate=rate,
                           decay_threshold=50, activation=activation)
    dfkm.U = pre_2.U
    dfkm.C = pre_2.C
    dfkm.W[1] = pre_1.W[1]
    dfkm.W[4] = pre_1.W[2]
    dfkm.b[1] = pre_1.b[1]
    dfkm.b[4] = pre_1.b[2]

    dfkm.W[2] = pre_2.W[1]
    dfkm.W[3] = pre_2.W[2]
    dfkm.b[2] = pre_2.b[1]
    dfkm.b[3] = pre_2.b[2]
    dfkm.train()
    name = DIR_NAME + 'dfkm.mat'
    save_mat(dfkm.U, y, name)
    print(cluster_acc_by_U(dfkm.U, y)[0], cal_metric(dfkm.U, y))


def train_AE(X, y, k, i):
    '''
    :param X: Training Set, n * d
    :param y: Training Labels, n * 1
    :param k: Amount of clusters
    :param i: Used to generate the name of result file, see line 100
    :return: (purity, NMI)
    '''
    lam2 = 10 ** -2
    delta = 0.1
    rate = 0.001
    activation = TANH
    print('-----------------start pre-training 1-----------------')
    pre_1 = AutoEncoder(X, y, k, [1024, 512, 1024], max_iter=5, delta=delta,
                        lam2=lam2, file_name=PARAMS_NAMES[JAFFE], read_params=False, rate=rate,
                        activation=activation)
    pre_1.train()
    print('-----------------start pre-training 2-----------------')
    pre_2 = AutoEncoder(pre_1.H, y, k, [512, 300, 512], max_iter=5, delta=delta,
                        lam2=lam2, file_name=PARAMS_NAMES[JAFFE], read_params=False, rate=rate,
                        activation=activation)
    pre_2.train()
    print('-----------------start training-----------------')
    ae = AutoEncoder(X, y, k, [1024, 512, 300, 512, 1024], max_iter=35, delta=delta,
                     lam2=lam2, file_name=PARAMS_NAMES[JAFFE], read_params=False, rate=rate,
                     decay_threshold=50, activation=activation)
    ae.W[1] = pre_1.W[1]
    ae.W[4] = pre_1.W[2]
    ae.b[1] = pre_1.b[1]
    ae.b[4] = pre_1.b[2]

    ae.W[2] = pre_2.W[1]
    ae.W[3] = pre_2.W[2]
    ae.b[2] = pre_2.b[1]
    ae.b[3] = pre_2.b[2]
    ae.train()
    name = DIR_NAME + 'ae_' + str(i) + '.mat'
    # return train_baseline(ae.H, y, k, 10, KM)
    km = KMeans(k)
    y_pred = km.fit_predict(ae.H, y)
    p, mi = cal_metric2(y_pred, y, k)
    scio.savemat(name, {'y_predicted': y_pred, 'y': y})
    return p, mi


if __name__ == '__main__':
    if not os.path.exists(DIR_NAME):
        os.makedirs(DIR_NAME)
    import warnings
    warnings.filterwarnings('ignore')
    X, y = get_images(JAFFE)
    X = X.astype(np.float64)
    X /= np.max(X)
    cluster_count = 10
    options = {RFKM_GAMMA: 1, RFKM_SIGMA: 1, KKM_GAMMA: 100}
    train_DFKM(X, y, cluster_count)
    print(train_baseline(X, y, cluster_count, 10, KM, DIR_NAME, train_AE, options))
    print(train_baseline(X, y, cluster_count, 10, FKM, DIR_NAME, train_AE, options))
    print(train_baseline(X, y, cluster_count, 10, KKM, DIR_NAME, train_AE, options))
    print(train_baseline(X, y, cluster_count, 10, RFKM, DIR_NAME, train_AE, options))
    print(train_baseline(X, y, cluster_count, 10, AE, DIR_NAME, train_AE, options))
