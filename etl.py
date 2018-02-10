# coding=utf-8
import h5py
import numpy as np
import pandas as pd

class ETL:
    """Extract Transform Load class for all data operations pre model inputs.
    Data is read in generative way to allow for large datafiles and low memory utilisation"""

    def __init__(self, filename_in, filename_out, batch_size, x_window_size,
                 y_window_size, y_col, x_col, filter_cols, target_percent_change, train_test_split):
        self.filename_in = filename_in
        self.filename_out = filename_out
        self.batch_size = batch_size
        self.x_window_size = x_window_size
        self.y_window_size = y_window_size
        self.x_col = x_col
        self.y_col = y_col
        self.filter_cols = filter_cols
        self.target_percent_change = target_percent_change
        self.train_test_split = train_test_split

    def clean_data(self):
        """Clean and Normalize the data in batches `batch_size` at a time"""
        data = pd.read_csv(self.filename_in, index_col=0)

        if self.filter_cols:
        # Remove any columns from data that we don't need by getting the difference between cols and filter list
            rm_cols = set(data.columns) - set(self.filter_cols)
            for col in rm_cols:
                del data[col]

        # Convert y-predict column name to numerical index
        y_col = list(data.columns).index(self.y_col)
        x_col = list(data.columns).index(self.x_col)

        # normalize data and save min, max values of each column
        # self.data_min, self.data_max, normalized_data = self.min_max_normalize(data)

        num_rows = len(data)
        x_data = []
        y_data = []

        skip_first_n = (num_rows - self.x_window_size - self.y_window_size) % self.batch_size + 1
        print('> While cleaning data, not use the first {} data for batch size alignment'.format(skip_first_n))
        print('> Total Batch num: {}'.format((num_rows - skip_first_n - self.x_window_size - self.y_window_size + 1)/self.batch_size))
        i = skip_first_n
        dist = dict()
        while (i + self.x_window_size + self.y_window_size) <= num_rows:
            x_window_data = data[i:(i + self.x_window_size)]

            # normalize data
            _, _, norm_x_window_data = self.min_max_normalize(x_window_data)

            latest_x = data[(i + self.x_window_size -1):(i + self.x_window_size)]
            y_window_data = data[(i + self.x_window_size):(i + self.x_window_size + self.y_window_size)]
            y = self.is_n_percent_up(latest_x.values[-1, x_col], y_window_data.values[:, y_col], self.target_percent_change)

            # for checking y's distribution
            if y in dist.keys(): dist[y] += 1
            else: dist[y] = 0

            x_data.append(norm_x_window_data.values)
            y_data.append(y)
            i += 1

        print("[data distribution]")
        print('> {}'.format(dist))
        print("> num of 0 / total = {}".format(dist[0] / sum(dist.values())))

        # Convert from list to 3 dimensional numpy array [windows, window_val, val_dimension]
        x_np_arr = np.array(x_data)
        y_np_arr = np.array(y_data)

        # calculate num of train data
        nrows = len(x_data)
        ntrain = int(self.train_test_split * nrows)
        steps_per_epoch = int(ntrain / self.batch_size)
        ntrain = steps_per_epoch * self.batch_size

        # Random Shuffle ONLY FOR training data
        p = np.random.permutation(ntrain)
        p = np.append(p, list(i for i in range(ntrain, len(x_np_arr))))
        x_np_arr, y_np_arr = x_np_arr[p], y_np_arr[p]

        return (x_np_arr, y_np_arr)

    def create_clean_datafile(self):
        """Incrementally save a datafile of clean data ready for loading straight into model"""
        print('> Creating x & y data files...')

        with h5py.File(self.filename_out, 'w') as hf:
            x, y = self.clean_data()
            dset_x = hf.create_dataset('x', shape=x.shape)
            dset_y = hf.create_dataset('y', shape=y.shape)
            dset_x[:] = x
            dset_y[:] = y

    def generate_clean_data(self, start_index, end_index):

        with h5py.File(self.filename_out, 'r') as hf:
            if start_index < 0:
                start_index += hf['x'].shape[0]
            if end_index < 0:
                end_index += hf['x'].shape[0]
            i = start_index
            while True:
                if i >= end_index:
                    print("generate_clean_data : initialize start index to {}".format(start_index))
                    i = start_index

                data_x = hf['x'][i:i + self.batch_size]
                data_y = hf['y'][i:i + self.batch_size]

                i += self.batch_size
                yield (data_x, data_y)

    def min_max_normalize(self, data, data_min=pd.DataFrame(), data_max=pd.DataFrame()):
        """Normalize a Pandas dataframe using column-wise min-max normalization
        (can use custom min, max if desired)"""
        if data_min.empty: data_min = data.min()
        if data_max.empty: data_max = data.max()
        data_normalized = (data - data_min) / (data_max - data_min)
        return (data_min, data_max, data_normalized)

    def is_n_percent_up(self, x: float, y: list, n: float):

        cnt = 0
        n = n * 0.01
        for cur_y in y:
            if (cur_y - x) / x >= n:
                cnt += 1

        if cnt > 0:
            cnt = 1

        return cnt
