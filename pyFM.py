# pip install git+https://github.com/coreylynch/pyFM
from pyfm import pylibfm
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

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
    # 'gender',

    'occupation',
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

df = pd.read_csv('./data/input/train.csv')

record = df[FEATS].to_dict(orient='records')
label  = df[LABEL]

preprocess_records(record)

v = DictVectorizer()
X = v.fit_transform(record)
y = np.array(label, dtype=np.float64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

fm = pylibfm.FM(num_iter=20, verbose=True, task="regression", initial_learning_rate=1e-4, learning_rate_schedule="optimal")
fm.fit(X_train, y_train)

print("Test MSE: %.4f" % mean_squared_error(y_test, fm.predict(X_test)))

