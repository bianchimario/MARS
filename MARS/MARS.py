import random
import numpy as np

class MARS:
    def __init__(self, shapelet_length, num_permutations, same_start=False, seed=None):
        self.shapelet_length = shapelet_length
        self.num_permutations = num_permutations
        self.same_start = same_start
        self.seed = seed

    def get_shapelets(self, time_series):
        if self.seed is not None:
            random.seed(self.seed)

        num_dimensions = len(time_series[0])
        max_window_length = len(time_series[0][0])

        if self.shapelet_length > max_window_length:
            raise ValueError("Shapelet length is greater than the length of the time series dimensions. "
                             "Please specify a valid shapelet length.")

        sliding_windows = []

        for _ in range(self.num_permutations):
            shapelet = []

            for dim_idx in range(num_dimensions):
                window_length = self.shapelet_length
                if not self.same_start:
                    window_indices = random.randint(0, max_window_length - window_length)
                else:
                    window_indices = 0
                window = [ts[dim_idx][window_indices:window_indices + window_length] for ts in time_series]
                shapelet.append(window)

            sliding_windows.append(shapelet)

        return sliding_windows

    def calculate_shapelet_distance(self, shapelet, time_series):
        min_distance = float('inf')

        for shapelet_window, time_series_window in zip(shapelet, time_series):
            total_distance = 0.0

            for dim in range(len(shapelet_window)):
                dimension_distance = np.linalg.norm(shapelet_window[dim] - time_series_window[dim])
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
