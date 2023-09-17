import numpy as np
import random

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


def get_distance(time_series, shapelet): # distance between univariate time series and shapelet
    max_idx = len(time_series) - len(shapelet)
    min_dist = float('inf')
    
    for i in range(0, max_idx):
        distance = np.linalg.norm(time_series[i:i+len(shapelet)] - shapelet) # euclidean distance
        if distance < min_dist:
            min_dist = distance

    return distance


def transform(time_series_dataset, shapelets):
    dims = len(time_series_dataset[0])

    distances_dataset = []

    for ts in time_series_dataset:
        ts_distances = [] # list of distances from a time series to all the shapelets
        for shapelet in shapelets:
            tot_dist = 0 # distance from ts to single shapelet
            for dim in range(0,dims):
                dim_dist = get_distance(ts[dim],shapelet[dim]) # distance on each dimension
                tot_dist += dim_dist
            distances_dataset.append(ts_distances)

    return distances_dataset