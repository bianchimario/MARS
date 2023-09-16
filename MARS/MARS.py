import random
import numpy as np

class MARS:
    def __init__(self, shapelet_length, num_permutations, same_start=False, seed=None):
        self.shapelet_length = shapelet_length
        self.num_permutations = num_permutations
        self.same_start = same_start
        self.seed = seed

    def get_shapelets(self, time_series):
        '''
        Extracts multivariate shapelets
        '''
        if self.seed is not None:
            random.seed(self.seed)

        num_dimensions = len(time_series[0])
        #max_window_length = len(time_series[0][0])
        max_window_length = min([len(e) for e in time_series[0]]) # length of the shortest dimension

        if self.shapelet_length > max_window_length:
            raise ValueError("Shapelet length is greater than the length of the time series dimensions. "
                             "Please specify a valid shapelet length.")

        sliding_windows = [] # each element of sliding_windows represents a shapelet for a specific permutation

        for _ in range(self.num_permutations): # for every shapelet created from a time series
            shapelet = [] # it represents one set of sliding windows for all dimensions of the MTS

            for dim_idx in range(num_dimensions): # for every dimension
                window_length = self.shapelet_length
                
                window_indices = random.randint(0, max_window_length - window_length) # index from zero to last possible position
                
                window = [ts[dim_idx][window_indices:window_indices + window_length] for ts in time_series]
                shapelet.append(window)

            sliding_windows.append(shapelet)

        return sliding_windows


    def calculate_shapelet_distance(self, shapelet, time_series):
        min_distance = float('inf')
        shapelet_length = len(shapelet[0][0])

        for start_pos in range(len(time_series[0][0]) - shapelet_length + 1):
            total_distance = 0.0

            # Ensure that you stay within the valid range for each dimension.
            for dim in range(len(shapelet[0])):
                dimension_distance = np.linalg.norm(
                    shapelet[0][dim] - time_series[0][dim][start_pos:start_pos + shapelet_length]
                )
                total_distance += dimension_distance

            if total_distance < min_distance:
                min_distance = total_distance

        return min_distance




    def get_distances(self, shapelets, time_series_dataset):
        distances = []

        for time_series in time_series_dataset:
            distances_to_shapelets = []

            for shapelet in shapelets:
                min_distance = self.calculate_shapelet_distance(shapelet, time_series)
                distances_to_shapelets.append(min_distance)

            distances.append(distances_to_shapelets)

        return distances
