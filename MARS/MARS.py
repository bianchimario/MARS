import numpy as np
import random
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import Parallel, delayed


class MARS(BaseEstimator, TransformerMixin):
    def __init__(self, num_shapelets, max_len, min_len, async_limit=None, seed=None, 
                 indexes=False, shapelet_indexes = True, n_jobs=1):
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
            #print('Calculating distances for TS #', idx)
            ts_distances = []
            ts_indexes = []

            for shapelet in self.shapelets:
                if self.indexes:
                    dist, indexes = self.get_distance_multi_and_indexes(ts, shapelet)
                    ts_indexes.append(indexes)
                else:
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
            return distances_dataset, indexes_dataset
        else:
            return distances_dataset

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
                min_start_idx = random.randint(0, self.max_len - random_length)
                max_start_idx = min_start_idx + self.async_limit
                for dim in range(0, dims):
                    start_idx = random.randint(min_start_idx, max_start_idx)
                    single_shapelet.append(ts[dim][start_idx:start_idx + random_length])
                shapelets.append(single_shapelet)

            elif self.async_limit <= 0:
                random_length = random.randint(self.min_len, self.max_len)
                start_idx = random.randint(0, self.max_len - random_length)
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
        # Create a sliding window view of the time_series for efficient calculation
        time_series_windows = np.lib.stride_tricks.sliding_window_view(time_series, shapelet_len)
        # Calculate distances for all windows simultaneously
        distances = np.linalg.norm(time_series_windows - shapelet, axis=1)
        min_dist = np.min(distances)
        return min_dist

    def get_distance_multi(self, multivariate_time_series, multivariate_shapelet):
        '''
        Distance from a multivariate time series and a multivariate shapelet.
        The distance is intended as the sum of the distances on each dimension.
        '''
        tot_dist = 0
        for dim in range(len(multivariate_shapelet)):
            dim_dist = self.get_distance(multivariate_time_series[dim], multivariate_shapelet[dim])
            tot_dist += dim_dist
        return tot_dist

    def get_distance_and_index(self, time_series, shapelet):
        '''
        Same as get_distance(), but stores the index.
        '''
        shapelet_len = len(shapelet)
        time_series_windows = np.lib.stride_tricks.sliding_window_view(time_series, shapelet_len)
        distances = np.linalg.norm(time_series_windows - shapelet, axis=1)
        min_dist = np.min(distances)
        min_dist_idx = np.argmin(distances)

        return min_dist, min_dist_idx

    def get_distance_multi_and_indexes(self, multivariate_time_series, multivariate_shapelet):
        '''
        Same as get_distances_multi(), but stores the indexes.
        '''
        tot_dist = 0
        indexes_list = []
        for dim in range(len(multivariate_shapelet)):
            dim_dist, idx = self.get_distance_and_index(multivariate_time_series[dim], multivariate_shapelet[dim])
            tot_dist += dim_dist
            indexes_list.append(idx)
        return tot_dist, indexes_list

# Usage example
# mars = MARS(num_shapelets=10, max_len=10, min_len=5, async_limit=None, seed=42, indexes=False)
# mars.fit(X_train)
# transformed_data = mars.transform(X_train)



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
