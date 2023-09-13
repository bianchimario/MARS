import awkward
import random
import numpy as np
from scipy.stats import entropy


class MARS:
    def __init__(self, num_shapelets, window_length, threshold=None, top_k=None, seed=None):
        """
        Initialize the MARS classifier.

        Args:
        - num_shapelets (int): Number of shapelets to extract.
        - window_length (int): Length of sliding windows for each dimension.
        - threshold (float, optional): Distance threshold for shapelet matching.
        - top_k (int, optional): Number of top shapelets to select based on information gain.
        - seed (int, optional): Seed for random number generation.
        """
        self.num_shapelets = num_shapelets
        self.window_length = window_length
        self.threshold = threshold
        self.top_k = top_k
        self.seed = seed
        self.shapelets = None

    def generate_random_sliding_windows(self, time_series_dimensions, num_permutations):
        """
        Generate a specified number of random sliding windows for each dimension in a multivariate time series.

        Args:
        - time_series_dimensions (list of lists): The multivariate time series, where each list represents a dimension.
        - num_permutations (int): Number of random sliding windows to extract.

        Returns:
        - list of lists: A list of lists representing the random sliding windows.
        """
        if self.seed is not None:
            random.seed(self.seed)
        
        num_dimensions = len(time_series_dimensions)
        sliding_windows = []

        for _ in range(num_permutations):
            # Generate random start indices for each dimension
            combination = [
                random.randrange(0, len(dim) - self.window_length + 1)
                for dim in time_series_dimensions
            ]
            window = [
                dim[start_idx:start_idx + self.window_length]
                for dim, start_idx in zip(time_series_dimensions, combination)
            ]
            sliding_windows.append(window)

        return sliding_windows

    def extract_multivariate_shapelets(self, X, num_shapelets):
        """
        Extract a specified number of shapelets from all the Multivariate Time Series.

        Args:
        - X (list of lists): List of multivariate time series.
        - num_shapelets (int): Number of random shapelets to extract from each time series.

        Returns:
        - candidates (list of lists): A list of all the candidate shapelets obtained.
        """
        candidates = []
        for ts in X:
            list_of_dims = [dim for dim in ts]
            shapelets = self.generate_random_sliding_windows(list_of_dims, num_shapelets)
            candidates.extend(shapelets)
            
        return candidates

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

        for i in range(len(time_series[0]) - self.window_length + 1):
            subsequence = [
                dim[i:i + self.window_length]
                for dim in time_series
            ]
            distance = self.euclidean_distance(shapelet, subsequence)

            if distance < min_distance:
                min_distance = distance

        return min_distance

    def calculate_distances_for_shapelets(self, shapelets, time_series_dataset):
        """
        Calculate the minimum distance between each shapelet and all possible subsequences in a multivariate time series dataset.

        Args:
        - shapelets (list of lists): List of shapelets.
        - time_series_dataset (list of lists): List of multivariate time series.

        Returns:
        - distances (ndarray): Minimum distances for each shapelet with respect to each time series in the dataset.
        """
        distances = []

        for shapelet in shapelets:
            shapelet_distances = []

            for time_series in time_series_dataset:
                distance = self.calculate_distance(shapelet, time_series)
                shapelet_distances.append(distance)

            distances.append(shapelet_distances)

        return np.array(distances).flatten()

    def is_shapelet_match(self, shapelet, time_series):
        """
        Check if a shapelet matches a time series based on a distance threshold.

        Args:
        - shapelet (list of lists): The shapelet.
        - time_series (list of lists): The multivariate time series.

        Returns:
        - bool: True if the shapelet matches the time series, False otherwise.
        """
        for i in range(len(time_series[0]) - self.window_length + 1):
            subsequence = [
                dim[i:i + self.window_length]
                for dim in time_series
            ]
            distance = self.euclidean_distance(shapelet, subsequence)

            if distance <= self.threshold:
                return True

        return False

    def calculate_entropy(self, y):
        """
        Calculate the entropy of a list of class labels (y).

        Args:
        - y (list): List of class labels.

        Returns:
        - float: Entropy value.
        """
        class_probabilities = np.bincount(y) / len(y)
        return entropy(class_probabilities, base=2)

    def calculate_information_gain(self, shapelet, X, y):
        """
        Calculate Information Gain for a shapelet.

        Args:
        - shapelet (list of lists): The shapelet.
        - X (list of lists): List of multivariate time series.
        - y (list): List of class labels.

        Returns:
        - float: Information Gain value.
        """
        entropy_original = self.calculate_entropy(y)

        matches = [i for i, ts in enumerate(X) if self.is_shapelet_match(shapelet, ts)]
        no_matches = [i for i in range(len(X)) if i not in matches]

        entropy_after_split = (
            (len(matches) / len(X)) * self.calculate_entropy(y[matches]) +
            (len(no_matches) / len(X)) * self.calculate_entropy(y[no_matches])
        )

        information_gain = entropy_original - entropy_after_split
        return information_gain

    def rank_shapelets_by_information_gain(self, shapelets, X, y):
        """
        Rank shapelets by Information Gain.

        Args:
        - shapelets (list of lists): List of shapelets.
        - X (list of lists): List of multivariate time series.
        - y (list): List of class labels.

        Returns:
        - ranked_shapelets (list of lists): List of ranked shapelets.
        """
        information_gains = [self.calculate_information_gain(shapelet, X, y) for shapelet in shapelets]
        shapelet_info_pairs = list(zip(shapelets, information_gains))
        ranked_shapelets = [shapelet for shapelet, _ in sorted(shapelet_info_pairs, key=lambda x: x[1], reverse=True)]

        if self.top_k is not None:
            ranked_shapelets = ranked_shapelets[:self.top_k]

        return ranked_shapelets
    
    # Yield successive n-sized chunks from l.
    def divide_chunks(self, l, n):
        # Looping until length l
        for i in range(0, len(l), n):
            yield l[i:i + n]


    def fit_transform(self, X, y):
        """
        Fit the MARS classifier to the training data and transform it.

        Args:
        - X (list of lists): List of multivariate time series.
        - y (list): List of class labels.

        Returns:
        - X_transformed (ndarray): Transformed training data.
        """
        if self.seed is not None:
            np.random.seed(self.seed)

        self.X = X
        self.y = y

        shapelets = self.extract_multivariate_shapelets(X, self.num_shapelets)
        self.shapelets = self.rank_shapelets_by_information_gain(shapelets, X, y)

        X_transformed = self.calculate_distances_for_shapelets(self.shapelets, X)

        # Reshape the transformed data
        num_time_series = len(X)  
        chunk_size = len(X_transformed) // num_time_series
        X_transformed = list(self.divide_chunks(X_transformed, chunk_size))

        return X_transformed


    def transform(self, X):
        """
        Transform new data using the learned shapelets.

        Args:
        - X (list of lists): List of multivariate time series.

        Returns:
        - X_transformed (list of lists): Transformed data.
        """
        if self.shapelets is None:
            raise ValueError("The shapelets have not been learned. Please call fit_transform() first.")

        # Calculate distances for shapelets
        X_transformed = self.calculate_distances_for_shapelets(self.shapelets, X)

        # Reshape the transformed data
        num_time_series = len(X)  
        chunk_size = len(X_transformed) // num_time_series
        X_transformed = list(self.divide_chunks(X_transformed, chunk_size))

        return X_transformed
    


