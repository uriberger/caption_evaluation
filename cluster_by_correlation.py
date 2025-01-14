from rating_datasets.polaris_dataset import PolarisDataset
import scipy.stats as stats
import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

dataset = PolarisDataset()
dataset.load()
selected_metrics = ['BLIP2Score', 'polos', 'PACScore', 'Exact noun overlap', 'BLEU1', 'Fuzzy verb overlap', 'BLEU4', 'CIDEr', 'ROUGE', 'RefCLIPScore']

human_rating_list = []
metric_to_score_list = {metric: [] for metric in selected_metrics}
metric_to_missing_inds = {metric: set() for metric in selected_metrics}

for image_data in dataset.data['train'].values():
    for caption_data in image_data['captions']:
        for human_rating in caption_data['human_ratings']:
            for metric in selected_metrics:
                if metric not in caption_data['automatic_metrics'] or np.isnan(caption_data['automatic_metrics'][metric]):
                    metric_to_missing_inds[metric].add(len(human_rating_list))
                    metric_to_score_list[metric].append(np.nan)
                else:
                    metric_to_score_list[metric].append(caption_data['automatic_metrics'][metric])
            human_rating_list.append(human_rating)

n = len(selected_metrics)
corr_mat = np.zeros((n, n))
for i in range(n):
    for j in range(i, n):
        metric1 = selected_metrics[i]
        metric2 = selected_metrics[j]
        joint_missing_inds = metric_to_missing_inds[metric1].union(metric_to_missing_inds[metric2])
        metric1_score_list = [metric_to_score_list[metric1][i] for i in range(len(metric_to_score_list[metric1])) if i not in joint_missing_inds]
        metric2_score_list = [metric_to_score_list[metric2][i] for i in range(len(metric_to_score_list[metric2])) if i not in joint_missing_inds]
        cur_corr = stats.pearsonr(metric1_score_list, metric2_score_list).statistic
        corr_mat[i, j] = cur_corr
        corr_mat[j, i] = cur_corr

best_silhouette_score = -2
best_labels = None
best_cluster_num = 0
for cluster_num in range(2, 9):
    sc = SpectralClustering(n_clusters=N ,affinity='precomputed')
    cur_score = silhouette_score(corr_mat, sc.fit_predict(corr_mat))
    if cur_score > best_silhouette_score:
        best_silhouette_score = cur_score
        best_labels = sc.labels_
        best_cluster_num = cluster_num

print(f'Best Cluster num: {best_cluster_num}')
print(f'Clusters: {[[selected_metrics[i] for i in range(len(selected_metrics)) if best_labels[i] == j] for j in range(best_cluster_num)]}')
