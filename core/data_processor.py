import numpy as np
import pandas as pd


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, filename, split, cols, predicted_col=0):
        df = pd.read_csv(filename)
        train, validate, test = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])
        i_split = int(len(df) * split)
        self.data_train = df.get(cols).values[:i_split]
        self.data_test = df.get(cols).values[i_split:]
        self.len_train = len(self.data_train)
        self.len_test = len(self.data_test)
        self.len_train_windows = None
        self.predicted_col = predicted_col

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        data_windows = []
        for i in range(self.len_test - seq_len):
            data_windows.append(self.data_test[i:i + seq_len])

        data_windows = np.array(data_windows).astype(float)
        data_windows_original = np.copy(data_windows)
        data_windows = self.normalise_windows(data_windows, single_window=False) if normalise else data_windows

        x = data_windows[:, :-1]
        y = data_windows[:, -1, [self.predicted_col]]
        x_o = data_windows_original[:, :-1]
        y_o = data_windows_original[:, -1, [self.predicted_col]]
        return x, y, x_o, y_o

    def get_train_data(self, seq_len, normalise):
        '''
        Create x, y train data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise use generate_training_window() method.
        '''
        data_x = []
        data_y = []
        for i in range(self.len_train - seq_len):
            x, y = self._next_window(i, seq_len, normalise)
            data_x.append(x)
            data_y.append(y)
        return np.array(data_x), np.array(data_y)

    def generate_train_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while True:  # i < (self.len_train - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_train - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    # yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(self.data_train, i, seq_len, normalise)
                assert len(x) == seq_len - 1
                assert len(y) == 1
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            assert batch_size == len(x_batch)
            assert batch_size == len(y_batch)
            yield np.array(x_batch), np.array(y_batch)

    def generate_test_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while True:  # i < (self.len_test - seq_len):
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                if i >= (self.len_test - seq_len):
                    # stop-condition for a smaller final batch if data doesn't divide evenly
                    # yield np.array(x_batch), np.array(y_batch)
                    i = 0
                x, y = self._next_window(self.data_test, i, seq_len, normalise)
                assert len(x) == seq_len - 1
                assert len(y) == 1
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            assert batch_size == len(x_batch)
            assert batch_size == len(y_batch)
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, df, i, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = df[i:i + seq_len]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        assert len(x) == seq_len - 1
        assert np.array_equal(x, window[0:seq_len - 1])
        y = window[-1, [self.predicted_col]]
        assert len(y) == 1
        assert len(x) + len(y) == seq_len
        return x, y

    def normalise_windows(self, window_data, single_window=False):
        '''Normalise window with a base value of zero'''
        normalised_data = []
        window_data = [window_data] if single_window else window_data
        for window in window_data:
            normalised_window = []
            for col_i in range(window.shape[1]):
                normalised_col = [((float(p) / float(window[0, col_i])) - 1) for p in window[:, col_i]]
                normalised_window.append(normalised_col)
            normalised_window = np.array(
                normalised_window).T  # reshape and transpose array back into original multidimensional format
            normalised_data.append(normalised_window)
        return np.array(normalised_data)
