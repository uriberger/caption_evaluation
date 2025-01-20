from rating_datasets.polaris_dataset import PolarisDataset
import numpy as np
import math
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

def select_predictor_metrics(normalize=False, direction='forward', tol=0.0001):
    train_set = PolarisDataset()
    train_set.load()

    del train_set.data['validation']
    del train_set.data['test']

    all_metrics = train_set.get_all_metrics()

    N = sum([sum([sum([len(caption_data['human_ratings']) for caption_data in image_data['captions']]) for image_data in dataset_data.values()]) for dataset_data in train_set.data.values()])
    X = np.zeros((N, len(all_metrics)))
    y = np.zeros(N)

    if normalize:
        metric_to_min_val = [math.inf for _ in all_metrics]
        metric_to_max_val = [(-1)*math.inf for _ in all_metrics]

    cur_sample_ind = 0
    for dataset_data in train_set.data.values():
        for image_data in dataset_data.values():
            for caption_data in image_data['captions']:
                for human_rating in caption_data['human_ratings']:
                    y[cur_sample_ind] = human_rating
                    for metric_ind, metric in enumerate(all_metrics):
                        if metric in caption_data['automatic_metrics']:
                            val = caption_data['automatic_metrics'][metric]
                            X[cur_sample_ind, metric_ind] = val
                            if normalize:
                                if val < metric_to_min_val[metric_ind]:
                                    metric_to_min_val[metric_ind] = np.float64(val)
                                if val > metric_to_max_val[metric_ind]:
                                    metric_to_max_val[metric_ind] = np.float64(val)
                        else:
                            X[cur_sample_ind, metric_ind] = np.nan
                    cur_sample_ind += 1

    if normalize:
        X = X - metric_to_min_val
        X = X / [metric_to_max_val[i] - metric_to_min_val[i] for i in range(len(all_metrics))]

    X = np.nan_to_num(X) # Convert all nans to zero

    reg = LinearRegression()
    sfs = SequentialFeatureSelector(reg, direction=direction, tol=tol)
    sfs.fit(X, y)
    support = sfs.get_support()
    selected_metrics = [all_metrics[i] for i in range(len(all_metrics)) if support[i]]
    X = X[:, support]
    reg = LinearRegression().fit(X, y)
    ensemble_weights = {selected_metrics[i]: reg.coef_[i] for i in range(len(selected_metrics))}

    if normalize:
        metric_to_min_val = [metric_to_min_val[i] for i in range(len(all_metrics)) if support[i]]
        metric_to_max_val = [metric_to_max_val[i] for i in range(len(all_metrics)) if support[i]]
        return ensemble_weights, metric_to_min_val, metric_to_max_val
    else:
        return ensemble_weights
