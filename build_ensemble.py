from rating_datasets.polaris_dataset import PolarisDataset
import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

def select_predictor_metrics():
    train_set = PolarisDataset()
    train_set.load()

    del train_set.data['validation']
    del train_set.data['test']

    all_metrics = train_set.get_all_metrics()

    N = sum([sum([sum([len(caption_data['human_ratings']) for caption_data in image_data['captions']]) for image_data in dataset_data.values()]) for dataset_data in train_set.data.values()])
    X = np.zeros((N, len(all_metrics)))
    y = np.zeros(N)

    cur_sample_ind = 0
    for dataset_data in train_set.data.values():
        for image_data in dataset_data.values():
            for caption_data in image_data['captions']:
                for human_rating in caption_data['human_ratings']:
                    y[cur_sample_ind] = human_rating
                    for metric_ind, metric in enumerate(all_metrics):
                        if metric in caption_data['automatic_metrics']:
                            X[cur_sample_ind, metric_ind] = caption_data['automatic_metrics'][metric]
                    cur_sample_ind += 1

    reg = LinearRegression()
    sfs = SequentialFeatureSelector(reg, direction='forward', tol=0.0001)
    sfs.fit(X, y)
    support = sfs.get_support()
    selected_metrics = [all_metrics[i] for i in range(len(all_metrics)) if support[i]]
    X = X[:, support]
    reg = LinearRegression().fit(X, y)
    ensemble_weights = {selected_metrics[i]: reg.coef_[i] for i in range(len(selected_metrics))}

    return ensemble_weights
