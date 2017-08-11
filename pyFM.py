# pip install git+https://github.com/coreylynch/pyFM
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
import numpy as np
import pandas as pd
from collections import OrderedDict

def preprocess_records(records):
    for d in records:
        for k in d:
            if k == 'user' or k =='item':
                v = str(d[k])
                d.update({k:v})

FEATS = [
    'user',
    'item',
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

df_train = pd.read_csv('./data/input/train.csv')[0:1000]
df_valid = pd.read_csv('./data/input/train.csv')[1000:1300]

record_train = df_train[FEATS].to_dict(orient='records')
record_valid = df_valid[FEATS].to_dict(orient='records')

preprocess_records(record_train)
preprocess_records(record_valid)

y_train = np.array(df_train[LABEL])
y_valid = np.array(df_valid[LABEL], dtype=np.double)

v = DictVectorizer()
x_train = v.fit_transform(record_train)
x_valid = v.fit_transform(record_valid)

print(x_train.toarray())
print(x_valid.toarray())

# data_train = [
#     {"user": "1", "item": "5", "age": 19},
#     {"user": "2", "item": "43", "age": 33},
#     {"user": "3", "item": "20", "age": 55},
#     {"user": "4", "item": "10", "age": 20},
# ]
# y_train = np.array([1, 0, 1, 0])
# v = DictVectorizer()
# x_train = v.fit_transform(data_train)
# print(x_train.toarray())
# fm = pylibfm.FM()
fm = pylibfm.FM(num_iter=10, verbose=True, task="regression", initial_learning_rate=1e-4, learning_rate_schedule="optimal")
fm.fit(x_train, y_train)
fm.predict(v.transform(x_valid))