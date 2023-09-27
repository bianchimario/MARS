import numpy as np
from scipy.spatial import distance
import random
from numba import njit, prange


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

def get_distance(time_series, shapelet): # distance between univariate time series and shapelet
    shapelet_len = len(shapelet)
    max_idx = len(time_series) - len(shapelet)
    min_dist = float('inf')
    
    for i in range(0, max_idx):
        dist = distance.euclidean(time_series[i:i+shapelet_len], shapelet) # euclidean distance
        #dist = np.linalg.norm(time_series[i:i+shapelet_len] - shapelet)
        if dist < min_dist:
            min_dist = dist

    return min_dist


def get_distances(time_series_dataset, shapelets):
    dims = len(time_series_dataset[0])

    distances_dataset = []

    for idx,ts in enumerate(time_series_dataset):
        print('Calculating distances for TS #', idx)
        ts_distances = [] # list of distances from a time series to all the shapelets
        for shapelet in shapelets:
            tot_dist = 0 # distance from ts to single shapelet
            for dim in range(0,dims):
                dim_dist = get_distance(ts[dim],shapelet[dim]) # distance on each dimension
                tot_dist += dim_dist
            ts_distances.append(tot_dist)
        distances_dataset.append(ts_distances)

    return distances_dataset


# --------------------------- Calculating distances (non-asynchronous shapelets) ---------------------------

@njit(parallel=True)
def get_distance_sync(multivariate_time_series, dimensions, shapelet):
    shapelet_length = len(shapelet[0])
    max_idx = len(multivariate_time_series[0]) - shapelet_length
    min_dist = float('inf')

    flat_shapelet = np.ravel(shapelet) # flattening array
    for idx in prange(0, max_idx):
        subsequence = np.ravel([multivariate_time_series[dim][idx:idx+shapelet_length] for dim in prange(0,dimensions)])
        flat_subsequence = np.ravel(subsequence) # flattening array
        #dist = distance.euclidean(flat_subsequence, flat_shapelet)
        dist = np.linalg.norm(flat_subsequence - flat_shapelet)
        if dist < min_dist:
            min_dist = dist

    return min_dist


@njit(parallel=True)
def get_multivariate_distances(time_series_dataset, shapelets):
    dims = len(shapelets[0])
    distances_dataset = []

    for idx,ts in enumerate(time_series_dataset):
        print('Calculating distances for TS #', idx)
        ts_distances = [] # list of distances from a time series to all the shapelets
        for shapelet in shapelets:
            dist = get_distance_sync(ts, dims, shapelet)
            ts_distances.append(dist)
        distances_dataset.append(ts_distances)
            
    return distances_dataset