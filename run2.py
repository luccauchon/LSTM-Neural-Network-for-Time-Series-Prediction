import datetime as dt
import math
import os

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, LSTM, Dense
from keras.models import Model

from core.data_processor import DataLoader
from core.utils import Timer

sequence_length = 20
batch_size = 32
input_timesteps = sequence_length - 1
cols = ["Close", 'Open', "Volume", 'High', 'Low']
predicted_col = 0
input_dim = len(cols)
epochs = 2
saved_dir = './saved_models/'

main_input = Input(shape=(input_timesteps, input_dim), dtype='float32', name='main_input')
x = LSTM(100, return_sequences=True, dropout=0.5, name='rnn_1')(main_input)
x = LSTM(100, dropout=0.5, name='rnn_2')(x)
x = Dense(1)(x)
main_output = x
model = Model(inputs=[main_input], outputs=[main_output])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[1.])
model.compile(optimizer='adam', loss='mse', loss_weights=[1.])
print(model.summary())

data = DataLoader('./data/sp500.csv', split=0.85, cols=cols, predicted_col=predicted_col)
# x, y = data.get_train_data(seq_len=sequence_length, normalise=True)

timer = Timer()
timer.start()
print('[Model] Training Started')
print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

save_filename = os.path.join(saved_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
callbacks = [
    EarlyStopping(monitor='val_loss', patience=2),
    ModelCheckpoint(filepath=save_filename, monitor='val_loss', save_best_only=True)
]
# model.fit(x, y, epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose = 1)
steps_per_epoch = math.ceil((data.len_train - sequence_length) / batch_size)
steps_per_epoch_val = math.ceil((data.len_test - sequence_length) / batch_size)
model.fit_generator(
    generator=data.generate_train_batch(seq_len=sequence_length, batch_size=batch_size, normalise=True),
    steps_per_epoch=steps_per_epoch, epochs=epochs, callbacks=callbacks, workers=0, verbose=2,
    validation_data=data.generate_test_batch(seq_len=sequence_length, batch_size=batch_size, normalise=True),
    validation_steps=steps_per_epoch_val)
model.save(save_filename)
print('[Model] Training Completed. Model saved as %s' % save_filename)
timer.stop()


def predict_sequences_multiple(model_, data_, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    print('[Model] Predicting Sequences Multiple...')
    prediction_seqs = []
    for i in range(int(len(data_) / prediction_len)):
        curr_frame = data_[i * prediction_len]
        predicted = []
        for j in range(prediction_len):
            y_pred = model_.predict(curr_frame[np.newaxis, ...])
            assert 1 == y_pred.shape[0]
            assert 1 == y_pred.shape[1]
            predicted.append(y_pred[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        # plt.legend()
    plt.show()


x_test, y_test, x_test_original, y_test_original = data.get_test_data(seq_len=sequence_length, normalise=True)
predictions = predict_sequences_multiple(model_=model, data_=x_test, window_size=sequence_length,
                                         prediction_len=sequence_length)

plot_results_multiple(predictions, y_test, sequence_length)


def predict_point_by_point(model_, data_):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('[Model] Predicting Point-by-Point...')
    predicted = model_.predict(data_)
    predicted = np.reshape(predicted, (predicted.size,))
    return predicted


predictions3 = predict_point_by_point(model, x_test)


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


plot_results(predictions3, y_test)


def predict_sequence_full(model_, data_, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    print('[Model] Predicting Sequences Full...')
    curr_frame = data_[0]
    predicted = []
    for i in range(len(data_)):
        y_pred = model_.predict(curr_frame[np.newaxis, ...])
        predicted.append(y_pred[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 2], predicted[-1], axis=0)
    return predicted


predictions3 = predict_sequence_full(model, x_test, sequence_length)
plot_results(predictions3, y_test)
