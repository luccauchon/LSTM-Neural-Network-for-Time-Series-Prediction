import datetime as dt
import os

import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, LSTM, Dense
from keras.models import Model
from sklearn.model_selection import ShuffleSplit

from core.data_processor import DataLoader
from core.utils import Timer

sequence_length = 9
batch_size = 64
n_steps = sequence_length - 1
cols = ["Close"]  # , "Volume"]#, 'High', 'Low', 'Open']
predicted_col = 0
n_features = len(cols)
epochs = 1
nfolds = 1
saved_dir = './saved_models/'
train_test_split = 0.90
train_val_split = 0.20

main_input = Input(shape=(n_steps, n_features), dtype='float32', name='main_input')
x = LSTM(100, activation='relu', return_sequences=True, dropout=0.5, name='rnn_1.1')(main_input)
x = LSTM(100, activation='relu', return_sequences=True, dropout=0.5, name='rnn_1.2')(x)
x = LSTM(100, activation='relu', name='rnn_2')(x)
x = Dense(1)(x)
main_output = x
model = Model(inputs=[main_input], outputs=[main_output])
model.compile(optimizer='adam', loss='mse', loss_weights=[1.])
print(model.summary())

###############################################################################
df = pd.read_csv('./data/sp500.csv')
'''
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
df = data.FB.dropna()'''
###############################################################################

train_validate_df, test_df = np.split(df.sample(frac=1), [int(train_test_split * len(df))])
print('train/validate on %s elements, test on %s elements' % (len(train_validate_df), len(test_df)))
assert len(train_validate_df) > len(test_df)
X = np.arange(len(train_validate_df))
ss = ShuffleSplit(n_splits=nfolds, test_size=train_val_split, random_state=0)
folds = list(ss.split(X))
for j, (train_idx, val_idx) in enumerate(folds):
    assert len(train_idx) > len(val_idx)
    data = DataLoader(df, train_idx, val_idx, cols=cols, predicted_col=predicted_col)

    timer = Timer()
    timer.start()

    save_filename = os.path.join(saved_dir, '%s-F%s.weights.{epoch:02d}-{val_loss:.6f}.hdf5' % (
        dt.datetime.now().strftime('%Y%m%d-%H%M%S'), str(j)))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2),
        ModelCheckpoint(filepath=save_filename, monitor='val_loss', save_best_only=False)
    ]

    steps_per_epoch = (data.len_train - sequence_length) // batch_size
    steps_per_epoch_val = (data.len_val - sequence_length) // batch_size
    model.fit_generator(
        generator=data.generate_train_batch(seq_len=sequence_length, batch_size=batch_size, normalise=True),
        steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, workers=0, verbose=2,
        validation_data=data.generate_val_batch(seq_len=sequence_length, batch_size=batch_size, normalise=True),
        validation_steps=steps_per_epoch_val)

    timer.stop()


def invert_prediction(df_, y_pred_, index_at_p0, predicted_cols_):
    p0 = df_.loc[index_at_p0, predicted_cols_]
    y_pred_inverted_ = p0 * (y_pred_ + 1)
    return y_pred_inverted_


def predict_point_by_point(model_, x_):
    predicted = model_.predict(x_)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted[0]


def predict_sequences_multiple_pp(model_, x_, df_, indexes_, sequence_length_, n_steps_, n_features_, predicted_cols_):
    assert x_.shape[0] == 1
    predictions = list()
    real_values = list()
    for i in range(sequence_length_):
        y_pred_ = predict_point_by_point(model_, x_)
        y_pred_inverted_ = invert_prediction(df_, y_pred_, index_at_p0=indexes_[0], predicted_cols_=predicted_cols_)
        y_real_ = df_.loc[indexes_[1] - 1 + i, predicted_cols_]
        predictions.add(y_pred_inverted_)
        real_values.add(y_real_)

        x_ = x_.reshape(n_steps_, n_features_)
        x_ = x_[1:]
        x_ = np.insert(x_, n_steps_ - 1, y_pred_, axis=0)
        x_ = x_[np.newaxis, ...]
    return np.hstack((np.array(predictions), np.array(real_values)))


###############################################################################
data = DataLoader(df, None, test_df.index, cols=cols, predicted_col=predicted_col)
x_test, y_test, indexes = data.get_test_data(seq_len=sequence_length, normalise=True)
###############################################################################


###############################################################################
for x, y, index in zip(x_test, y_test, indexes):
    x = x[np.newaxis, ...]
    predict_sequences_multiple_pp(model, x, df, indexes, sequence_length_=sequence_length, n_steps_=n_steps,
                                  n_features_=n_features, predicted_cols_=cols[predicted_col])

###############################################################################


###############################################################################
mean_pct_error = 0
for x, y, index in zip(x_test, y_test, indexes):
    # print(df.iloc[index[0]:index[1],])
    x = x[np.newaxis, ...]
    y_pred = predict_point_by_point(model, x)
    y_pred_inverted = invert_prediction(df, y_pred, index_at_p0=index[0], predicted_cols=cols[predicted_col])
    y_real = df.loc[index[1] - 1, cols[predicted_col]]
    error = (y_real - y_pred_inverted)
    pct_error = error / y_real * 100
    mean_pct_error += pct_error
    # print(str(pct_error)+'% > error of '+str(error)+'  We want to predict (' + str(y_real)+')  we got ('+str(y_pred_converted)+')')
mean_pct_error /= len(x_test)
print(str(mean_pct_error) + '% mean error')
###############################################################################
