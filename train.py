# ipython3
import pandas as pd
import numpy as np

train = pd.read_csv('./data/input/train.csv')
test  = pd.read_csv('./data/input/test.csv')

FEATS = [
    # 'user_id',
    # 'item_id',
    # 'timestamp',
    # 'age',
    'gender',
    # 'occupation',
    # 'release_date',
    'Action',
    'Adventure',
    'Animation',
    'Childrens',
    'Comedy',
    'Crime',
    'Documentary',
    'Drama',
    'Fantasy',
    'Film_Noir',
    'Horror',
    'Musical',
    'Mystery',
    'Romance',
    'Sci_Fi',
    'Thriller',
    'War',
    'Western',
    ]
LABEL = 'rating'

slicing = slice(1, 1000)
x_train, y_train = np.array(train[FEATS])[slicing], np.array(train[LABEL])[slicing]
x_test,  y_test  = np.array(test[FEATS])[slicing],  np.array(test[LABEL])[slicing]

def fm(x, w_0, w, v):
    n = len(FEATS)
    w_1 = (x * w).sum()
    w_2 = 0
    for i in range(0, n):
        for j in range(i + 1, n):
            w_2 += v[i].dot(v[j]) * x[i] * x[j]
    return w_0 + w_1 + w_2

def model(x_train, w_0, w, v):
    y = []
    for x in x_train:
        y = fm(x, w_0, w, v)
    return np.array(y)

def fm_grad_w_0(x, w_0, w, v):
    return 1

def fm_grad_w_i(x, w_0, w, v, i):
    return x[i]

def fm_grad_v_i_f(x, w_0, w, v, i, f):
    n = len(FEATS)
    a = 0
    for j in range(n):
        a += v[j][f] * x[j]
    return x[i] * a - v[i][f] * x[i] ** 2

def loss(x_train, y_train, w_0, w, v, norm_w_0, norm_w, norm_v):
    return np.mean(
        (model(x_train, w_0, w, v) - y_train) ** 2 
        # normlization
        + w_0 * norm_w_0 ** 2 
        + np.sum(w * norm_w ** 2) 
        + np.sum(v * norm_v ** 2))

def loss_grad_w_0(x, y, w_0, w, v, norm_w_0, norm_w, norm_v):
    return (2 * (fm(x, w_0, w, v) - y)) * fm_grad_w_0(x, w_0, w, v) + 2 * norm_w_0 * w_0

def loss_grad_w_i(x, y, w_0, w, v, i, norm_w_0, norm_w, norm_v):
    return (2 * (fm(x, w_0, w, v) - y)) * fm_grad_w_i(x, w_0, w, v, i) + 2 * norm_w[i] * w[i]

def loss_grad_v_i_f(x, y, w_0, w, v, i, f, norm_w_0, norm_w, norm_v):
    return (2 * (fm(x, w_0, w, v) - y)) * fm_grad_v_i_f(x, w_0, w, v, i, f) + 2 * norm_v[i][f] * v[i][f]

def eval(epoch, iteration, x_train, y_train, x_test, y_test, model, params):
    def rmse(a, b):
        return np.sqrt(np.mean( (a - b) ** 2) )
    w_0, w, v = params
    rmse_train = rmse(y_train, model(x_train, w_0, w, v))
    rmse_test  = rmse(y_test,  model(x_test,  w_0, w, v))
    loss_train = loss(x_train,  y_train, w_0, w, v, norm_w_0, norm_w, norm_v)
    print('epoch:%d iter:%d train loss:%.2f train rmse:%.2f test rmse:%.2f' % (epoch, iteration, loss_train, rmse_train, rmse_test))

eta = 1e-5 # learning rate
k = 5 # dim of factorization
sigma = 0.5
n = len(FEATS)
w_0 = 0
w = np.zeros((n))
v = np.ones((n, k))
norm_w_0 = np.random.normal(0, sigma, 1)
norm_w   = np.random.normal(0, sigma, n)
norm_v   = np.random.normal(0, sigma, (n, k))


for epoch in range(50):
    for it in range(x_train.shape[0]):
        eval(epoch, it, x_train, y_train, x_test, y_test, model, params=[w_0, w, v])
        x = x_train[it]
        y = y_train[it]

        # w_0train
        w -= eta * loss_grad_w_0(x, y, w_0, w, v, norm_w_0, norm_w, norm_v)
        # print('loss w_0:{}'.format(loss_grad_w_0(x, y, w_0, w, v, norm_w_0, norm_w, norm_v)))

        # w_i
        for i in range(w.shape[0]):
            if x[i] != 0:
                w[i] -= eta * loss_grad_w_i(x, y, w_0, w, v, i, norm_w_0, norm_w, norm_v)
                # print('loss w[{}]:{}'.format(i, loss_grad_w_i(x, y, w_0, w, v, i, norm_w_0, norm_w, norm_v)))

                # v_{i,f}
                for j in range(v.shape[1]):
                    v[i][j] -= eta * loss_grad_v_i_f(x, y, w_0, w, v, i, j, norm_w_0, norm_w, norm_v)


