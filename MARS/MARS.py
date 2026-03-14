import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


class MARS(BaseEstimator, TransformerMixin):
    def __init__(self, num_shapelets, max_len, min_len, async_limit=None, seed=None, 
                 indexes=False, shapelet_indexes = True, n_jobs=-1):
        self.num_shapelets = num_shapelets # Number of shapelets to extract
        self.max_len = max_len # Max length of the shapelet (same for each dimension)
        self.min_len = min_len
        self.async_limit = async_limit # Maximum starting index difference for each dimension
        self.seed = seed
        self.indexes = indexes # To save the index where the shapelet is the closest to the time series
        self.shapelets = None
        self.n_jobs = n_jobs
        self.shapelet_indexes = shapelet_indexes # Indexes of the TS from which the shapelets have been extracted

# ---------------------- Main Functions ----------------------

    def fit(self, time_series_dataset):
        '''
        fit() gets random shapelets from the given TS dataset.
        '''
        if self.seed is not None:
            random.seed(self.seed)

        if self.shapelet_indexes == False:
            self.shapelets, _ = self.get_random_shapelets(time_series_dataset)
        
        else:
            self.shapelets, self.shapelet_indexes = self.get_random_shapelets(time_series_dataset)
        
        return self


    def transform(self, time_series_dataset):
        '''
        transform() calculates the distance from each TS to each shapelet.
        '''
        distances_dataset = []
        if self.indexes:
            indexes_dataset = []

        def process_ts(idx, ts):
            ts_distances = []
            ts_indexes = []
            if self.indexes:
                for shapelet in self.shapelets:
                    dist, indexes = self.get_distance_multi_and_indexes(ts, shapelet)
                    ts_distances.append(dist)
                    ts_indexes.append(indexes)
            else:
                for shapelet in self.shapelets:
                    dist = self.get_distance_multi(ts, shapelet)
                    ts_distances.append(dist)
            return ts_distances, ts_indexes

        # Use joblib to parallelize the computation for each time series
        results = Parallel(n_jobs=self.n_jobs)(delayed(process_ts)(idx, ts) for idx, ts in enumerate(time_series_dataset))

        for ts_distances, ts_indexes in results:
            distances_dataset.append(ts_distances)
            if self.indexes:
                indexes_dataset.append(ts_indexes)

        if self.indexes:
            return np.array(distances_dataset), indexes_dataset
        else:
            return np.array(distances_dataset)

# ---------------------- Getting the shapelets ----------------------

    def get_random_shapelets(self, time_series_dataset):
        if self.seed is not None:
            random.seed(self.seed)

        dims = len(time_series_dataset[0])
        max_possible_length = min([len(e) for e in time_series_dataset[0]]) # length of the shortest dimension

        if (self.max_len > max_possible_length) or (self.min_len > max_possible_length):
            raise ValueError("Shapelet length is greater than the length of the shortest dimension.")

        shapelets = []
        #random_ts = random.sample(range(0, len(time_series_dataset)), self.num_shapelets) # indexes of the time series that generate a shapelet without repetitions
        random_ts = [random.randint(0, len(time_series_dataset)-1) for _ in range(self.num_shapelets)] # indexes of the time series that generate a shapelet

        for idx in random_ts:
            ts = time_series_dataset[idx]

            if self.async_limit is None:
                single_shapelet = []
                random_length = random.randint(self.min_len, self.max_len)
                for dim in range(0, dims):
                    start_idx = random.randint(0, len(ts[dim]) - random_length)
                    single_shapelet.append(ts[dim][start_idx:start_idx + random_length])
                shapelets.append(single_shapelet)

            elif self.async_limit > 0:
                single_shapelet = []
                random_length = random.randint(self.min_len, self.max_len)
                min_start_idx = random.randint(0, max_possible_length - random_length - self.async_limit)
                max_start_idx = min_start_idx + self.async_limit
                for dim in range(0, dims):
                    start_idx = random.randint(min_start_idx, max_start_idx)
                    single_shapelet.append(ts[dim][start_idx:start_idx + random_length])
                shapelets.append(single_shapelet)

            elif self.async_limit <= 0:
                single_shapelet = []
                random_length = random.randint(self.min_len, self.max_len)
                start_idx = random.randint(0, max_possible_length - random_length)
                for dim in range(0, dims):
                    single_shapelet.append(ts[dim][start_idx:start_idx + random_length])
                shapelets.append(single_shapelet)

        return shapelets, random_ts

# ---------------------- Calculating distances ----------------------

    def get_distance(self, time_series, shapelet):
        '''
        Distance from a univariate time series to a univariate shapelet.
        '''
        shapelet_len = len(shapelet)
        windows = np.lib.stride_tricks.sliding_window_view(time_series, shapelet_len)
        diff = windows - shapelet
        # Work in squared space: argmin is preserved, sqrt only on the minimum
        sq_dists = np.einsum('ij,ij->i', diff, diff)
        return np.sqrt(np.min(sq_dists))

    def get_distance_multi(self, multivariate_time_series, multivariate_shapelet):
        '''
        Distance from a multivariate time series and a multivariate shapelet.
        The distance is intended as the sum of the distances on each dimension.
        '''
        shapelet_len = len(multivariate_shapelet[0])
        # windows: (dims, n_windows, shapelet_len)
        windows = np.array([
            np.lib.stride_tricks.sliding_window_view(multivariate_time_series[d], shapelet_len)
            for d in range(len(multivariate_shapelet))
        ])
        shapelet_arr = np.array(multivariate_shapelet)  # (dims, shapelet_len)
        diff = windows - shapelet_arr[:, np.newaxis, :]  # (dims, n_windows, shapelet_len)
        # sq_dists: (dims, n_windows) — sqrt only on the per-dim minimum
        sq_dists = np.einsum('ijk,ijk->ij', diff, diff)
        return float(np.sum(np.sqrt(np.min(sq_dists, axis=1))))

    def get_distance_and_index(self, time_series, shapelet):
        '''
        Same as get_distance(), but stores the index.
        '''
        shapelet_len = len(shapelet)
        windows = np.lib.stride_tricks.sliding_window_view(time_series, shapelet_len)
        diff = windows - shapelet
        sq_dists = np.einsum('ij,ij->i', diff, diff)
        min_idx = np.argmin(sq_dists)
        return np.sqrt(sq_dists[min_idx]), int(min_idx)

    def get_distance_multi_and_indexes(self, multivariate_time_series, multivariate_shapelet):
        '''
        Same as get_distances_multi(), but stores the indexes.
        '''
        shapelet_len = len(multivariate_shapelet[0])
        # windows: (dims, n_windows, shapelet_len)
        windows = np.array([
            np.lib.stride_tricks.sliding_window_view(multivariate_time_series[d], shapelet_len)
            for d in range(len(multivariate_shapelet))
        ])
        shapelet_arr = np.array(multivariate_shapelet)  # (dims, shapelet_len)
        diff = windows - shapelet_arr[:, np.newaxis, :]  # (dims, n_windows, shapelet_len)
        # Squared distances: einsum computes sum(diff²) along shapelet axis → (dims, n_windows)
        # We stay in squared space to find argmin: argmin(d²) == argmin(d), avoiding n_windows sqrt calls
        sq_dists = np.einsum('ijk,ijk->ij', diff, diff)
        # argmin is the same in squared or normal space, so we can extract the index directly
        indexes_list = list(np.argmin(sq_dists, axis=1))
        # sqrt only on the per-dim minimum (dims values), not on all windows
        tot_dist = float(np.sum(np.sqrt(np.min(sq_dists, axis=1))))
        return tot_dist, indexes_list




'''
Non-parallelized transform function:

    def transform(self, time_series_dataset):
        distances_dataset = []
        if self.indexes:
            indexes_dataset = []

        for idx, ts in enumerate(time_series_dataset):
            ts_distances = []
            ts_indexes = []

            for shapelet in self.shapelets:
                if self.indexes:
                    dist, indexes = self.get_distance_multi_and_indexes(ts, shapelet)
                    ts_indexes.append(indexes)
                else:
                    dist = self.get_distance_multi(ts, shapelet)

                ts_distances.append(dist)

            distances_dataset.append(ts_distances)

            if self.indexes:
                indexes_dataset.append(ts_indexes)

        if self.indexes:
            return distances_dataset, indexes_dataset
        else:
            return distances_dataset
'''
