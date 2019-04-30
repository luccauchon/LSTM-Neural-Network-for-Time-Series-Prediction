import fix_yahoo_finance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from core.data_processor import DataLoader
import math

data = yf.download(
    # tickers list or string as well
    tickers=["MSFT",'FB','IBM'],

    # use "period" instead of start/end
    # valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    # (optional, default is '1mo')
    period="max",

    # fetch data by interval (including intraday if period < 60 days)
    # valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    # (optional, default is '1d')
    interval="1d",

    # group by ticker (to access via data['SPY'])
    # (optional, default is 'column')
    group_by='ticker',

    # adjust all OHLC automatically
    # (optional, default is False)
    auto_adjust=True,

    # download pre/post regular market hours data
    # (optional, default is False)
    prepost=True
)

print (data.FB)
print (data.IBM)
print (data.MSFT)

train_test_split = 0.95
train_val_split = 0.15
nfolds = 1
batch_size=1
sequence_length=6
df = data.IBM
train_validate_df, test_df = np.split(df.sample(frac=1), [int(train_test_split * len(df))])
print('train/validate on %s elements, test on %s elements' % (len(train_validate_df), len(test_df)))
assert len(train_validate_df) > len(test_df)
X = np.arange(len(train_validate_df))
ss = ShuffleSplit(n_splits=nfolds, test_size=train_val_split, random_state=0)
folds = list(ss.split(X))
mean_x =0
mean_y=0
count=0
for j, (train_idx, val_idx) in enumerate(folds):
    assert len(train_idx) > len(val_idx)
    fdata = DataLoader(df, train_idx, val_idx, cols=["Close"], ipredicted_col=0)
    steps_per_epoch = (fdata.len_train - sequence_length) // batch_size
    for x_batch, y_batch in fdata.generate_train_batch(seq_len=sequence_length, batch_size=batch_size, normalise=True):
        #print (x_batch)
        mean_x += np.absolute(x_batch.mean())
        mean_y += np.absolute(y_batch.mean())
        count += 1
        if (count >steps_per_epoch):
            break
        #print (y_batch)

print (mean_x/count)
print (mean_y/count)