import numpy as np
from scipy.spatial import distance
import random

# ---------------------------------- Getting Shapelets ----------------------------------

def get_shapelets(time_series_dataset, num_shapelets, len_shapelets, async_shapelets=True, seed=None):
    if seed is not None:
        random.seed(seed)

    dims = len(time_series_dataset[0])
    max_length = min([len(e) for e in time_series_dataset[0]]) # length of the shortest dimension

    shapelets = []

    if len_shapelets>max_length:
        raise ValueError("Shapelet length is greater than the length of the shortest dimension.")
    
    for ts in time_series_dataset:
        
        if async_shapelets:
            for _ in range(0,num_shapelets): 
                single_shapelet = []
                for dim in range(0,dims): # async: new starting point for each dimension
                    start_idx = random.randint(0,len(ts[dim]) - len_shapelets)
                    single_shapelet.append(ts[dim][start_idx:start_idx+len_shapelets])
                shapelets.append(single_shapelet)

        else:
            for _ in range(0,num_shapelets):
                start_idx = random.randint(0, max_length - len_shapelets) # index from zero to last possible position
                single_shapelet = [ts[dim][start_idx:start_idx+len_shapelets] for dim in range(0,dims)]
                shapelets.append(single_shapelet)

    return shapelets


def get_random_shapelets(time_series_dataset, num_shapelets, max_len, min_len, async_shapelets=True, seed=None):
    if seed is not None:
        random.seed(seed)

    dims = len(time_series_dataset[0])
    max_possible_length = min([len(e) for e in time_series_dataset[0]]) # length of the shortest dimension

    if (max_len>max_possible_length) or (min_len>max_possible_length):
        raise ValueError("Shapelet length is greater than the length of the shortest dimension.")
    
    shapelets = []
    random_ts = random.sample(range(0, len(time_series_dataset)), num_shapelets) # indexes of the time series that generate a shapelet

    for idx in random_ts:
        ts = time_series_dataset[idx]

        if async_shapelets:
            single_shapelet = []
            for dim in range(0,dims): # async: new starting point for each dimension
                    random_length = random.randint(min_len,max_len)
                    start_idx = random.randint(0,len(ts[dim]) - random_length)
                    single_shapelet.append(ts[dim][start_idx:start_idx+random_length])
            shapelets.append(single_shapelet)

        else:
            random_length = random.randint(min_len,max_len)
            start_idx = random.randint(0, max_len - random_length) # index from zero to last possible position
            single_shapelet = [ts[dim][start_idx:start_idx+random_length] for dim in range(0,dims)]
            shapelets.append(single_shapelet)

    return shapelets


# ---------------------------------- Calculating distances ----------------------------------

def get_distance(time_series, shapelet):
    shapelet_len = len(shapelet)
    
    # Create a sliding window view of the time_series for efficient calculation
    time_series_windows = np.lib.stride_tricks.sliding_window_view(time_series, shapelet_len)
    
    # Calculate distances for all windows simultaneously
    distances = np.linalg.norm(time_series_windows - shapelet, axis=1)
    
    min_dist = np.min(distances)
    return min_dist


def get_distance_multi(multivariate_time_series, multivariate_shapelet):
    tot_dist = 0
    for dim in range(len(multivariate_shapelet)):
        dim_dist = get_distance(multivariate_time_series[dim], multivariate_shapelet[dim])
        tot_dist += dim_dist

    return tot_dist


def transform(time_series_dataset, shapelets):
    distances_dataset = []

    for idx,ts in enumerate(time_series_dataset):
        print('Calculating distances for TS #', idx)
        ts_distances = []
        for shapelet in shapelets:
            dist = get_distance_multi(ts, shapelet)
            ts_distances.append(dist)
        distances_dataset.append(ts_distances)

    return distances_dataset