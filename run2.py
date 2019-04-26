import datetime as dt
import math
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, LSTM, Dense
from keras.models import Model

from core.data_processor import DataLoader
from core.utils import Timer

sequence_length = 9
batch_size = 32
input_timesteps = sequence_length - 1
cols = ["Close", 'Open', "Volume"]
input_dim = len(cols)
epochs = 15
saved_dir = './saved_models/'

main_input = Input(shape=(input_timesteps, input_dim), dtype='float32', name='main_input')
x = LSTM(100, return_sequences=True, name='rnn_1')(main_input)
x = LSTM(100, name='rnn_2')(x)
x = Dense(1)(x)
main_output = x
model = Model(inputs=[main_input], outputs=[main_output])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy',loss_weights=[1.])
model.compile(optimizer='adam', loss='mse', loss_weights=[1.])
print(model.summary())

data = DataLoader('./data/sp500.csv', split=0.85, cols=cols)
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
