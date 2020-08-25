from deep_fuzzy_k_means import *
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from skfuzzy import cmeans


colors = [[116, 189, 255],
          [251, 229, 167],
          [255,255,224],
          [98,154,113],
          [28,159,205],
          [242, 242, 242],
          [76, 109, 172],
          [51, 152, 255],
          [211, 210, 215],
          [6,43,106],
          [166, 188, 111],
          [251, 189, 75],
          [60,179,113],
          [123,104,238],
          [248,249,57],
          [176, 117, 163],
          [53,46,132]]
dir_name = 'img_seg/human/'
colors = np.array(colors)
img = mpimg.imread('img_seg/human.bmp')
# img = mpimg.imread('img_seg/cars.bmp')
# img = mpimg.imread('img_seg/human.bmp')
# img = mpimg.imread('img_seg/dog.png')
# img = (img * 255).astype(np.uint8)
# img = mpimg.imread('/Users/hyzhang/Downloads/MSRC_ObjCategImageDatabase_v1/6_27_s.bmp')
K = [13]
for k in K:
    T = img.copy()
    print(img.shape)
    width = img.shape[0]
    height = img.shape[1]
    dim = img.shape[2]
    print(img[0][0])
    X = np.zeros((width*height, 3))
    index = 0
    for i in range(width):
        for j in range(height):
            X[index] = img[i][j]
            index += 1
    coef = np.max(X)
    X /= coef
    results = cmeans(X.T, k, m=3, error=0.01, maxiter=100)
    U = results[1].T
    index = 0
    for i in range(width):
        for j in range(height):
            center_index = np.argmax(U[index])
            T[i][j] = colors[center_index]
            index += 1
    plt.subplot(1, 2, 1)
    plt.imshow(T)
    C = T.copy()
    plt.imsave(fname=dir_name + 'human_fcm_' + str(k) + '.png', arr=T/255)
    gamma = 0.01
    delta = 0.1
    lam1 = 0.1
    lam2 = 0.01
    batch = 10000
    print('-----------------start pre-training 1-----------------')
    pre_1 = DeepFuzzyKMeans(X, None, k, [3, 3, 3], max_iter=5, gamma=gamma, delta=delta, lam1=lam1, lam2=lam2,
                            read_params=False, rate=0.001, batch=batch)
    # pre_1.U = U
    # pre_1.train()
    print('-----------------start pre-training 2-----------------')
    pre_2 = DeepFuzzyKMeans(X, None, k, [3, 3, 3], max_iter=5, gamma=gamma, delta=delta, lam1=lam1, lam2=lam2,
                            read_params=False, rate=0.001, batch=batch)
    # pre_2.U = pre_1.U
    # pre_2.train()
    print('-----------------train-----------------')
    dfkm = DeepFuzzyKMeans(X, None, k, [3, 3, 3, 3, 3], max_iter=15, gamma=gamma, delta=delta, lam1=lam1, lam2=lam2,
                           read_params=True, rate=0.001, batch=batch, file_name='img_seg_dfkm', decay_threshold=10)
    # dfkm.U = pre_2.U
    dfkm.C = pre_2.C
    dfkm.W[1] = pre_1.W[1]
    dfkm.b[1] = pre_1.b[1]
    dfkm.W[4] = pre_1.W[2]
    dfkm.b[4] = pre_1.b[2]
    dfkm.W[2] = pre_2.W[1]
    dfkm.b[2] = pre_2.b[1]
    dfkm.W[3] = pre_2.W[2]
    dfkm.b[3] = pre_2.b[2]
    # dfkm.train()
    # dfkm.write_params()
    index = 0
    for i in range(width):
        for j in range(height):
            center_index = (1 + np.argmax(dfkm.U[index])) % 13
            # center_index = np.argmax(dfkm.U[index])
            T[i][j] = colors[center_index]
            index += 1
    A = np.unique(dfkm.C, axis=1)
    print(A.shape)
    print((C == T).all())
    plt.subplot(1, 2, 2)
    plt.imshow(T)
    # plt.imsave(fname=dir_name + 'human_dfkm_' + str(k) + '.png', arr=T/255)
# plt.show()
