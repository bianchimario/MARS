import numpy as np
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


def get_random_shapelets(time_series_dataset, num_shapelets, max_len, min_len, async_limit=None, seed=None):
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

        if async_limit==None:
            single_shapelet = []
            random_length = random.randint(min_len,max_len)
            for dim in range(0,dims): # async: new starting point for each dimension
                start_idx = random.randint(0,len(ts[dim]) - random_length)
                single_shapelet.append(ts[dim][start_idx:start_idx+random_length])
            shapelets.append(single_shapelet)

        elif async_limit>0:
            single_shapelet = []
            random_length = random.randint(min_len,max_len)
            min_start_idx = random.randint(0, max_len - random_length)
            max_start_idx = min_start_idx + async_limit
            for dim in range(0,dims):
                start_idx = random.randint(min_start_idx, max_start_idx)
                single_shapelet.append(ts[dim][start_idx:start_idx+random_length])
            shapelets.append(single_shapelet)

        else: # async_limit<=0
            random_length = random.randint(min_len,max_len)
            start_idx = random.randint(0, max_len - random_length) # index from zero to last possible position
            single_shapelet = [ts[dim][start_idx:start_idx+random_length] for dim in range(0,dims)]
            shapelets.append(single_shapelet)

    return shapelets


def fit(time_series_dataset, num_shapelets, max_len, min_len, async_limit=None, seed=None):
    output = get_random_shapelets(time_series_dataset, num_shapelets, max_len, min_len, async_limit, seed)
    return output


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


def get_distance_and_index(time_series, shapelet):
    shapelet_len = len(shapelet)
    
    # Create a sliding window view of the time_series for efficient calculation
    time_series_windows = np.lib.stride_tricks.sliding_window_view(time_series, shapelet_len)
    
    # Calculate distances for all windows simultaneously
    distances = np.linalg.norm(time_series_windows - shapelet, axis=1)
    
    min_dist = np.min(distances) # get mimimum
    min_dist_idx = np.argmin(distances) # get index

    return min_dist,min_dist_idx


def get_distance_multi_and_indexes(multivariate_time_series, multivariate_shapelet):
    tot_dist = 0
    indexes_list = [] # list of indexes where there is the mimimum dist. between shapelet and MTS
    for dim in range(len(multivariate_shapelet)):
        dim_dist,idx = get_distance_and_index(multivariate_time_series[dim], multivariate_shapelet[dim])
        tot_dist += dim_dist
        indexes_list.append(idx)
    return tot_dist,indexes_list


def transform(time_series_dataset, shapelets, indexes=False):
    distances_dataset = []
    indexes_dataset = []

    if indexes==False:
        for idx,ts in enumerate(time_series_dataset):
            print('Calculating distances for TS #', idx)
            ts_distances = []
            for shapelet in shapelets:
                dist = get_distance_multi(ts, shapelet)
                ts_distances.append(dist)
            distances_dataset.append(ts_distances)

        return distances_dataset
    
    else: # save indexes
        for idx,ts in enumerate(time_series_dataset):
            print('Calculating distances for TS #', idx)
            ts_distances = []
            ts_indexes = []
            for shapelet in shapelets:
                dist,indexes = get_distance_multi_and_indexes(ts, shapelet)
                ts_distances.append(dist)
                ts_indexes.append(indexes)
            distances_dataset.append(ts_distances)
            indexes_dataset.append(ts_indexes)

        return distances_dataset,indexes_dataset