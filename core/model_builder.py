import keras
from keras.layers import Input, LSTM, Dense, Conv1D, MaxPooling1D, Flatten
from keras.models import Model


def build_lstm_univariate(input_shape):
    main_input = Input(shape=input_shape, dtype='float32', name='main_input')
    x = LSTM(100, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.0001),
             name='rnn_1.1')(main_input)
    x = LSTM(100, activation='relu', return_sequences=True, kernel_regularizer=keras.regularizers.l2(0.0001),
             name='rnn_1.2')(x)
    x = LSTM(100, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001), name='rnn_2')(x)
    x = Dense(1)(x)
    main_output = x

    model = Model(inputs=[main_input], outputs=[main_output])
    model.compile(optimizer='adam', loss='mse', loss_weights=[1.])

    print(model.summary())
    return model


def build_multi_headed_cnn_univariate(input_shape):
    n_steps = input_shape[0]
    n_features = input_shape[1]

    visible1 = Input(shape=(n_steps, n_features))
    cnn1 = Conv1D(filters=64, kernel_size=2, kernel_regularizer=keras.regularizers.l2(0.0001), activation='relu')(
        visible1)
    cnn1 = MaxPooling1D(pool_size=2)(cnn1)
    cnn1 = Flatten()(cnn1)

    visible2 = Input(shape=(n_steps, n_features))
    cnn2 = Conv1D(filters=64, kernel_size=2, kernel_regularizer=keras.regularizers.l2(0.0001), activation='relu')(
        visible2)
    cnn2 = MaxPooling1D(pool_size=2)(cnn2)
    cnn2 = Flatten()(cnn2)

    merge = keras.backend.concatenate([cnn1, cnn2])
    dense = Dense(50, activation='relu', kernel_regularizer=keras.regularizers.l2(0.0001))(merge)
    output = Dense(1)(dense)

    model = Model(inputs=[visible1, visible2], outputs=output)
    return model
