from sklearn.cluster import KMeans, SpectralClustering
import numpy as np
from scipy import stats
from sklearn.metrics import normalized_mutual_info_score
import scipy.io as scio
from fuzzy_k_means import FuzzyKMeans
from RobustFKM import RobustFKM
from kernel_k_means import KernelKMeans
# from robust_sparse_fkm import RobustSparseFKM


KM = 0
FKM = 1
SC = 2
AE = 3
RFKM = 4
KKM = 5
RSFKM = 6
FKM_GAMMA = 'fkm_gamma'
FKM_CONVERGE_ACC = 'fkm_convergence_ACC'
FKM_FILE_NAME = 'fkm_file_name'
SC_GAMMA = 'sc_gamma'
RFKM_GAMMA = 'rfkm_gamma'
RFKM_SIGMA = 'rfkm_sigma'
KKM_GAMMA = 'kkm_gamma'
KKM_KERNEL = 'kkm_kernel'


def cal_metric(U, y):
    return cal_acc(U, y), cal_nmi(U, y)


def cal_metric2(y_pre, y, c):
    return cal_acc2(y_pre, y, c), nmi(y, y_pre)


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
        if len(g) == 0:
            # print('-----')
            continue
        count += stats.mode(g).count[0]
    # print(count)
    return count / size


def cal_nmi(U, y):
    n = U.shape[0]
    y_predicted = []
    for i in range(n):
        y_predicted.append(np.argmax(U[i]))
    y_predicted = np.array(y_predicted)
    return normalized_mutual_info_score(y, y_predicted, average_method='arithmetic')


def nmi(y, y_predicted):
    return normalized_mutual_info_score(y, y_predicted, average_method='arithmetic')


def save_mat(U, labels, file_name=None):
    n, c = U.shape
    size = labels.shape[0]
    predict_y = []
    for i in range(n):
        predict_y.append(np.argmax(U[i]))
    predict_y = np.matrix(predict_y).reshape((size, 1)) + 1
    y = labels.reshape((size, 1))
    mat_name = 'dfkm.mat' if file_name is None else file_name
    scio.savemat(mat_name, {'y_predicted': predict_y, 'y': y})


def train_baseline(X, y, k, n, model, DIR_NAME, train_AE, options={}):
    pur = []
    accs = []
    nmis = []
    assert isinstance(options, dict)
    for i in range(n):
        print(i)
        if model == SC:
            if SC_GAMMA in options.keys():
                gamma = options[SC_GAMMA]
            else:
                gamma = 0.0001
            sc = SpectralClustering(k, gamma=gamma)
            y_pre = sc.fit_predict(X)
            p, mi = cal_metric2(y_pre, y, k)
            acc,freq = cluster_acc(y_pre, y)
            name = DIR_NAME + 'sc_' + str(i) + '.mat'
            scio.savemat(name, {'y_predicted': y_pre, 'y': y})
        elif model == KM:
            km = KMeans(n_clusters=k)
            y_pre = km.fit_predict(X)
            p, mi = cal_metric2(y_pre, y, k)
            acc,freq = cluster_acc(y_pre, y)
            name = DIR_NAME + 'km_' + str(i) + '.mat'
            scio.savemat(name, {'y_predicted': y_pre, 'y': y})
        elif model == FKM:
            if FKM_GAMMA in options.keys():
                gamma = options[FKM_GAMMA]
            else:
                gamma = 3
            if FKM_CONVERGE_ACC in options.keys():
                convergence_acc = options[FKM_CONVERGE_ACC]
            else:
                convergence_acc = 10 ** -2
            if FKM_FILE_NAME in options.keys():
                file_name = options[FKM_FILE_NAME]
            else:
                file_name = 'fkm'
            fkm = FuzzyKMeans(X, k, gamma, file_name, convergence_acc, print_msg=False)
            fkm.train()
            # fkm.print_max_prob(10)
            # print(cal_metric(fkm.U, y))
            p, mi = cal_metric(fkm.U, y)
            acc,freq = cluster_acc_by_U(fkm.U, y)
            name = DIR_NAME + 'fkm_' + str(i) + '.mat'
            save_mat(fkm.U, y, name)
        elif model == AE:
            p, acc, mi = train_AE(X, y, k, i)
        elif model == RFKM:
            if RFKM_GAMMA in options.keys():
                gamma = options[RFKM_GAMMA]
            else:
                gamma = 1
            if RFKM_SIGMA in options.keys():
                sigma = options[RFKM_SIGMA]
            else:
                sigma = 1
            rfkm = RobustFKM(X, y, k, gamma, 0.01, sigma, print_msg=False)
            rfkm.train()
            p, mi = cal_metric(rfkm.U, y)
            acc,freq = cluster_acc_by_U(rfkm.U, y)
            name = DIR_NAME + 'rfkm_' + str(i) + '.mat'
            save_mat(rfkm.U, y, name)
        elif model == KKM:
            if KKM_GAMMA in options.keys():
                gamma = options[KKM_GAMMA]
            else:
                gamma = 100
            if KKM_KERNEL in options.keys():
                kernel = options[KKM_KERNEL]
            else:
                kernel = KernelKMeans.GAUSSIAN
            kkm = KernelKMeans(X, k, gamma, kernel)
            y_pre = kkm.train()
            p, mi = cal_metric2(y_pre, y, k)
            acc,freq = cluster_acc(y_pre, y)
            name = DIR_NAME + 'kkm_' + str(i) + '.mat'
            scio.savemat(name, {'y_predicted': y_pre, 'y': y})
        elif model == RSFKM:
            rsfkm = RobustSparseFKM(X, k, 1)
            rsfkm.train()
            p, mi = cal_metric(rsfkm.U, y)
            acc,freq = cluster_acc_by_U(rsfkm.U, y)
            name = DIR_NAME + 'rsfkm_' + str(i) + '.mat'
            save_mat(rsfkm.U, y, name)
        else:
            raise Exception('No corresponding model!')
        pur.append(p)
        accs.append(acc)
        nmis.append(mi)
    return np.mean(pur), np.std(pur), np.mean(accs), np.std(accs), np.mean(nmis), np.std(nmis)


def cluster_acc_by_U(U, y):
    y_pre = np.argmax(U, axis=1)
    return cluster_acc(y_pre, y)


def cluster_acc(Y_pred, Y):
    from sklearn.utils.linear_assignment_ import linear_assignment
    # from scipy.optimize import linear_sum_assignment as linear_assignment
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max())+1
    w = np.zeros((D,D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i,j] for i,j in ind])*1.0/Y_pred.size, w
