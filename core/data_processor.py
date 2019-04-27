import keras
import numpy as np


class DataLoader():
    """A class for loading and transforming data for the lstm model"""

    def __init__(self, df, train_idx, val_idx, cols, predicted_col=0):
        self.df = df.get(cols)
        self.train_idx = train_idx
        self.val_idx = val_idx
        if self.train_idx is not None:
            self.len_train = len(self.train_idx)
        self.len_test = len(self.val_idx)
        self.predicted_col = predicted_col

    def get_test_data(self, seq_len, normalise):
        '''
        Create x, y test data windows
        Warning: batch method, not generative, make sure you have enough memory to
        load data, otherwise reduce size of the training split.
        '''
        x_batch = []
        y_batch = []
        indexes = []
        for b in self.val_idx:
            start_index = b
            end_index = start_index + seq_len
            if end_index > len(self.df) - 1:
                continue
            x, y = self._next_window(start_index, end_index, seq_len, normalise)
            x_batch.append(x)
            y_batch.append(y)
            indexes.append((start_index, end_index))

        return np.array(x_batch), np.array(y_batch), indexes

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
        while True:
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                while True:
                    try:
                        start_index = self.train_idx[i]
                    except:
                        i = 0
                        continue
                    end_index = start_index + seq_len
                    if end_index < len(self.df):
                        break
                    i += 1
                x, y = self._next_window(start_index, end_index, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            assert batch_size == len(x_batch)
            assert batch_size == len(y_batch)
            yield np.array(x_batch), np.array(y_batch)

    def generate_test_batch(self, seq_len, batch_size, normalise):
        '''Yield a generator of training data from filename on given list of cols split for train/test'''
        i = 0
        while True:
            x_batch = []
            y_batch = []
            for b in range(batch_size):
                while True:
                    try:
                        start_index = self.val_idx[i]
                    except:
                        i = 0
                        continue
                    end_index = start_index + seq_len
                    if end_index < len(self.df):
                        break
                    i += 1
                x, y = self._next_window(start_index, end_index, seq_len, normalise)
                x_batch.append(x)
                y_batch.append(y)
                i += 1
            assert batch_size == len(x_batch)
            assert batch_size == len(y_batch)
            yield np.array(x_batch), np.array(y_batch)

    def _next_window(self, start_index, end_index, seq_len, normalise):
        '''Generates the next data window from the given index location i'''
        window = self.df.values[start_index:end_index]
        window = self.normalise_windows(window, single_window=True)[0] if normalise else window
        x = window[:-1]
        assert x.shape[0] == seq_len - 1
        assert len(x.shape) == 2
        assert np.array_equal(x, window[0:seq_len - 1])
        y = window[-1, [self.predicted_col]]
        assert len(y.shape) == 1
        assert y.shape[0] == 1
        assert x.shape[0] + y.shape[0] == seq_len
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


class DataLoader2(keras.utils.Sequence):
    """2019-04-26"""

    def __init__(self):
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.img_ids) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        :param index:
        :return:
        """

        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        rows = [self.img_ids[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(rows)

        return x, y

    def on_epoch_end(self):
        """
        Updates indexes after each epoch.
        :return:
        """
        np.random.shuffle(self.indexes)

    def __data_generation(self, rows):
        """
        Generates data
        :param rows:
        :return:
        """
        start = time.time()
        if self.check_on: assert len(rows) == self.batch_size, 'len(rows)=' + str(
            len(rows)) + ' ==  len(batch_size)=' + str(self.batch_size)
        # Initialization
        x = np.ones([self.batch_size, self.img_w, self.img_h, self.nb_channel])
        y = np.zeros([self.batch_size, self.img_w, self.img_h, self.number_classes])

        # Generate data
        for i, img_id in enumerate(rows):
            None
        end = time.time()
        LOG.debug('Got batch of ' + str(len(x)) + ' elements in ' + str(end - start) + ' seconds, ' + str(
            (end - start) / len(x)) + ' seconds per image.\nrows=' + str(rows))
        return x, y
