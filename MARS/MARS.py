import awkward
import random
import numpy as np
from scipy.stats import entropy


class MARS:
    def __init__(self, num_shapelets, shapelet_length, seed=None):
        """
        Initialize the MARS classifier.

        Args:
        - num_shapelets (int): Number of shapelets to extract.
        - shapelet_length (int): Length of shapelets.
        - seed (int, optional): Seed for random number generation.
        """
        self.num_shapelets = num_shapelets
        self.shapelet_length = shapelet_length
        self.seed = seed
        self.shapelets = None

    def generate_random_sliding_windows(self, time_series, num_permutations):
        """
        Generate a specified number of random sliding windows for each dimension in a multivariate time series.

        Args:
        - time_series (list of lists): The multivariate time series, where each list represents a dimension.
        - num_permutations (int): Number of random sliding windows to extract.

        Returns:
        - list of lists: A list of lists representing the random sliding windows.
        """
        if self.seed is not None:
            random.seed(self.seed)

        num_dimensions = len(time_series)
        sliding_windows = []

        for _ in range(num_permutations):
            # Generate random start index for all dimensions
            start_indices = [
                random.randrange(0, len(dim) - self.shapelet_length + 1)
                for dim in time_series
            ]

            window = [
                dim[start_idx:start_idx + self.shapelet_length]
                for dim, start_idx in zip(time_series, start_indices)
            ]

            sliding_windows.append(window)

        return sliding_windows

    def euclidean_distance(self, a, b):
        """
        Calculate the Euclidean distance between two multivariate time series subsequences (a and b).

        Args:
        - a, b (list of lists): Multivariate time series subsequences.

        Returns:
        - float: Euclidean distance between a and b.
        """
        squared_distance = sum(np.sum((np.array(a[i]) - np.array(b[i]))**2) for i in range(len(a)))
        return np.sqrt(squared_distance)

    def calculate_distance(self, shapelet, time_series):
        """
        Calculate the minimum distance between a shapelet and all possible subsequences in a multivariate time series.

        Args:
        - shapelet (list of lists): The shapelet.
        - time_series (list of lists): The multivariate time series.

        Returns:
        - float: Minimum distance between the shapelet and any subsequence in the time series.
        """
        min_distance = float('inf')

        for i in range(len(time_series[0]) - self.shapelet_length + 1):
            subsequence = [dim[i:i + self.shapelet_length] for dim in time_series]
            distance = self.euclidean_distance(shapelet, subsequence)

            if distance < min_distance:
                min_distance = distance

        return min_distance

    def get_shapelets(self, X):
        """
        Get a specified number of shapelets from all the Multivariate Time Series.

        Args:
        - X (list of lists): List of multivariate time series.

        Returns:
        - candidates (list of lists): A list of all the candidate shapelets obtained.
        """
        candidates = []
        for ts in X:
            shapelets = self.generate_random_sliding_windows(ts, self.num_shapelets)
            candidates.extend(shapelets)

        return candidates

    def get_distances(self, X, shapelets):
        """
        Calculate the minimum distance between each shapelet and all possible subsequences in a multivariate time series dataset.

        Args:
        - X (list of lists): List of multivariate time series.
        - shapelets (list of lists): List of shapelets.

        Returns:
        - distances (ndarray): Minimum distances for each shapelet with respect to each time series in the dataset.
        """
        distances = []

        for shapelet in shapelets:
            shapelet_distances = []

            for time_series in X:
                distance = self.calculate_distance(shapelet, time_series)
                shapelet_distances.append(distance)

            distances.append(shapelet_distances)

        return np.array(distances).flatten()

    


