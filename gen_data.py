# ipython3

import pandas as pd
import datetime

occupation_dict = {'administrator':1, 'artist':2, 'doctor':3, 'educator':4, 'engineer':5, 'entertainment':6, 'executive':7, 'healthcare':8, 'homemaker':9, 'lawyer':10, 'librarian':11, 'marketing':12, 'none':13, 'other':14, 'programmer':15, 'retired':16, 'salesman':17, 'scientist':18, 'student':19, 'technician':20, 'writer':21} 

rate_cols = ['user', 'item', 'rating', 'timestamp']
user_cols = ['user', 'age', 'gender', 'occupation', 'zip_code']
item_cols = ['item', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film_Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci_Fi', 'Thriller', 'War', 'Western']

train_paths = ['./data/raw/ml-100k/u%d.base' % i for i in range(1,3)]
test_paths  = ['./data/raw/ml-100k/u%d.test' % i for i in range(1,3)]
user_path = './data/raw/ml-100k/u.user'
item_path = './data/raw/ml-100k/u.item'

def read_rate(rate_paths):
    user = pd.read_csv(user_path, sep='|', names=user_cols)
    item = pd.read_csv(item_path, sep='|', names=item_cols)
    rate = pd.concat([pd.read_csv(i, sep='\t', names=rate_cols) for i in rate_paths])

    rate = rate.join(user.set_index('user'), on='user', how='left')
    rate = rate.join(item.set_index('item'), on='item', how='left')

    rate['gender'] = rate['gender'].map({'F':0, 'M':1})
    rate['occupation'] = rate['occupation'].map(occupation_dict)
    rate['release_date'] = rate['release_date'].map(lambda x : int(datetime.datetime.strptime(x, '%d-%b-%Y')) if type(x) == str() else -9999)
    rate = rate[[
        'user',
        'item',
        'rating',
        'timestamp',
        'age',
        'gender',
        'occupation',
        # 'zip_code',
        'release_date',
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
        'Western']]
    return rate

train = read_rate(train_paths)
train.to_csv('./data/input/train.csv', index=False)

test = read_rate(test_paths)
test.to_csv('./data/input/test.csv', index=False)

