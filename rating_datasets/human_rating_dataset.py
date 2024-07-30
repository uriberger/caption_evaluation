import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
import pickle
from tqdm import tqdm
import csv

class HumanRatingDataset:
    def __init__(self):
        self.data = {}

    def get_name(self):
        return ''

    def dump(self, dump_file=None):
        if dump_file is None:
            dump_file = f'{self.get_name()}_data.pkl'
        with open(dump_file, 'wb') as fp:
            pickle.dump(self.data, fp)

    def load(self):
        dump_file = f'{self.get_name()}_data.pkl'
        with open(dump_file, 'rb') as fp:
            self.data = pickle.load(fp)
    
    def collect_data(self):
        # Each dataset implements its own data collection
        return
    
    def get_image(self, split_name, image_data):
        # Get image object given the image data and split name
        return
    
    def get_file_path(self, split_name, image_data):
        # Get image path given the image data and split name
        return
    
    def clean_temp_files(self):
        return
    
    def get_data_list(self, split_name):
        image_ids = []
        image_paths = []
        caption_inds = []
        references = []
        candidates = []
        for image_id, image_data in self.data[split_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                cur_refs = image_data['references']
                cur_cand = caption_data['caption']
                references.append(cur_refs)
                candidates.append(cur_cand)
                file_path = self.get_file_path(split_name, image_data)
                image_paths.append(file_path)
                image_ids.append(image_id)
                caption_inds.append(caption_ind)

        return image_ids, image_paths, caption_inds, references, candidates
    
    def log_scores(self, split_name, image_ids, caption_inds, metric_name, scores):
        for image_id, caption_ind, score in zip(image_ids, caption_inds, scores):
            self.data[split_name][image_id]['captions'][caption_ind]['automatic_metrics'][metric_name] = float(score)

    def compute_metrics(self, compute_clip_image_score=False, overwrite=False):
        for split in self.data.keys():
            self.compute_metrics_for_split(split, compute_clip_image_score, overwrite)

    def compute_metrics_for_split(self, split_name, compute_clip_image_score, overwrite):
        if overwrite or ('METEOR' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics']):
            self.compute_coco_metrics_for_split(split_name)
        if overwrite or ('CLIPScore' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics']):
            self.compute_clipscore_for_split(split_name)
        if overwrite or ('Exact noun overlap' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics']):
            self.compute_content_overlap_metrics_for_split(split_name)
        if overwrite or ('BLIP2Score' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics']):
            self.compute_blip2score_for_split(split_name)
        if overwrite or ('polos' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics']):
            self.compute_polos_for_split(split_name)
        if overwrite or ('MPNetScore' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics']):
            self.compute_mpnet_score_for_split(split_name)
        if overwrite or ('PACScore' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics']):
            self.compute_pacscore_for_split(split_name)
        if compute_clip_image_score and (overwrite or ('CLIPImageScore' not in list(self.data[split_name].values())[0]['captions'][0]['automatic_metrics'])):
            self.compute_clip_image_score_for_split(split_name)

    def compute_coco_metrics_for_split(self, split_name):
        from metrics.coco_metrics import compute_coco_metrics

        image_ids, _, caption_inds, references, candidates = self.get_data_list(split_name)
        metric_name_to_scores = compute_coco_metrics(candidates, references)

        # Log scores
        for metric_name, scores in metric_name_to_scores.items():
            self.log_scores(split_name, image_ids, caption_inds, metric_name, scores)
    
    def compute_clipscore_for_split(self, split_name):
        from metrics.clipscore import compute_clipscore
        image_ids, image_paths, caption_inds, references, candidates = self.get_data_list(split_name)
        
        clipscores, refclipscores = compute_clipscore(candidates, references, image_paths)
        self.log_scores(split_name, image_ids, caption_inds, 'CLIPScore', clipscores)
        self.log_scores(split_name, image_ids, caption_inds, 'RefCLIPScore', refclipscores)
    
    def compute_pacscore_for_split(self, split_name):
        from metrics.pacscore import compute_pacscore
        image_ids, image_paths, caption_inds, references, candidates = self.get_data_list(split_name)

        pac_scores, refpac_scores = compute_pacscore(candidates, references, image_paths)

        self.log_scores(split_name, image_ids, caption_inds, 'PACScore', pac_scores)
        self.log_scores(split_name, image_ids, caption_inds, 'RefPACScore', refpac_scores)
    
    def compute_content_overlap_metrics_for_split(self, split_name):
        from metrics.content_overlap_metrics import compute_content_overlap_metrics

        image_ids, _, caption_inds, references, candidates = self.get_data_list(split_name)
        scores = compute_content_overlap_metrics(candidates, references)

        for metric_name, metric_scores in scores.items():
            self.log_scores(split_name, image_ids, caption_inds, metric_name, metric_scores)
    
    def compute_polos_for_split(self, split_name):
        from metrics.polos import compute_polos

        image_ids, image_paths, caption_inds, references, candidates = self.get_data_list(split_name)
        scores = compute_polos(candidates, references, image_paths)

        self.log_scores(split_name, image_ids, caption_inds, 'polos', scores)

    def compute_clip_image_score_for_split(self, split_name):
        from metrics.clip_image_score import compute_clip_image_score
        image_ids, image_paths, caption_inds, _, candidates = self.get_data_list(split_name)
        
        scores = compute_clip_image_score(candidates, image_paths)

        self.log_scores(split_name, image_ids, caption_inds, 'CLIPImageScore', scores)
    
    def compute_mpnet_score_for_split(self, split_name, agg_method='mean'):
        from metrics.mpnet_score import compute_mpnet_score
        image_ids, _, caption_inds, references, candidates = self.get_data_list(split_name)
        
        scores = compute_mpnet_score(candidates, references, agg_method)
        
        self.log_scores(split_name, image_ids, caption_inds, 'MPNetScore', scores)

    def compute_blip2score_for_split(self, split_name):
        from metrics.blip2score import compute_blip2score
        image_ids, image_paths, caption_inds, _, candidates = self.get_data_list(split_name)
        
        scores = compute_blip2score(candidates, image_paths)

        self.log_scores(split_name, image_ids, caption_inds, 'BLIP2Score', scores)

    def get_all_metrics(self, sort_by_type=False):
        metrics_in_dataset = list(set([x for dataset_data in self.data.values() for image_data in dataset_data.values() for caption_data in image_data['captions'] for x in caption_data['automatic_metrics'].keys()]))
        metrics_in_dataset = [x for x in metrics_in_dataset if not x.startswith('SPICE_')]
        if sort_by_type:
            all_metrics = ['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'CIDEr', 'Exact noun overlap', 'Exact verb overlap', 'METEOR', 'ROUGE', 'Fuzzy noun overlap', 'Fuzzy verb overlap', 'SPICE', 'MPNetScore', 'BLIP2Score', 'CLIPImageScore', 'CLIPScore', 'PACScore', 'RefCLIPScore', 'RefPACScore', 'polos']
        else:
            all_metrics = ['Exact noun overlap', 'Fuzzy noun overlap', 'Exact verb overlap', 'Fuzzy verb overlap', 'CLIPScore', 'RefCLIPScore', 'METEOR', 'PACScore', 'ROUGE', 'RefPACScore', 'SPICE', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'BLIP2Score', 'CIDEr', 'CLIPImageScore', 'MPNetScore', 'polos']
            all_metrics.sort()

        return [x for x in all_metrics if x in metrics_in_dataset]

    def predict_with_ensemble_weights(self, metric_res, ensemble_weights):
        prediction = 0
        for metric_name, weight in ensemble_weights.items():
            if metric_name not in metric_res or np.isnan(metric_res[metric_name]):
                continue
            prediction += weight * metric_res[metric_name]
        return prediction

    def compute_correlation(self, plot=True, ensemble_weights=None, dataset_name=None, rating_column='human_ratings'):
        all_metrics = self.get_all_metrics()
        if ensemble_weights is not None:
            all_metrics.append('ensemble')
        human_rating_list = []
        metric_to_score_list = {metric: [] for metric in all_metrics}
        metric_to_missing_inds = {metric: set() for metric in all_metrics}
        if dataset_name is None:
            dataset_values = self.data.values()
        else:
            dataset_values = [self.data[dataset_name]]
        for dataset_data in dataset_values:
            for image_data in dataset_data.values():
                for caption_data in image_data['captions']:
                    for human_rating in caption_data[rating_column]:
                        for metric in all_metrics:
                            if metric == 'ensemble':
                                continue
                            if metric not in caption_data['automatic_metrics'] or np.isnan(caption_data['automatic_metrics'][metric]):
                                metric_to_missing_inds[metric].add(len(human_rating_list))
                                metric_to_score_list[metric].append(np.nan)
                            else:
                                metric_to_score_list[metric].append(caption_data['automatic_metrics'][metric])
                        if ensemble_weights is not None:
                            metric_to_score_list['ensemble'].append(self.predict_with_ensemble_weights(caption_data['automatic_metrics'], ensemble_weights))
                        human_rating_list.append(human_rating)

        self.compute_mutual_correlation(metric_to_score_list, metric_to_missing_inds, plot)
        return self.compute_correlation_with_human_ratings(human_rating_list, metric_to_score_list, metric_to_missing_inds)
    
    def compute_correlation_with_human_ratings(self, human_rating_list, metric_to_score_list, metric_to_missing_inds):
        all_metrics = list(metric_to_score_list.keys())
        
        corr_type_to_func = {'pearson': stats.pearsonr, 'spearman': stats.spearmanr, 'kendall_b': stats.kendalltau, 'kendall_c': lambda x,y: stats.kendalltau(x, y, variant='c')}
        corr_type_to_res = {}
        metric_to_corrs = {metric: {} for metric in all_metrics}
        for corr_type, corr_func in corr_type_to_func.items():
            metric_to_corr = {}
            for metric in all_metrics:
                cur_human_rating_list = [human_rating_list[i] for i in range(len(human_rating_list)) if i not in metric_to_missing_inds[metric]]
                cur_metric_score_list = [metric_to_score_list[metric][i] for i in range(len(metric_to_score_list[metric])) if i not in metric_to_missing_inds[metric]]
                cur_corr = float(corr_func(cur_human_rating_list, cur_metric_score_list).statistic)
                metric_to_corr[metric] = cur_corr
                metric_to_corrs[metric][corr_type] = cur_corr

            res = [(metric, metric_to_corr[metric]) for metric in all_metrics]
            res.sort(key=lambda x:x[1], reverse=True)
            corr_type_to_res[corr_type] = res

        with open('corr_with_human_ratings.csv', 'w') as fp:
            my_writer = csv.writer(fp)
            corr_types = ['pearson', 'spearman', 'kendall_b', 'kendall_c']
            my_writer.writerow(['metric'] + corr_types)
            metric_corrs_list = sorted(list(metric_to_corrs.items()), key=lambda x:x[1]['pearson'])
            for metric, corr_values in metric_corrs_list:
                my_writer.writerow([metric] + ['%.1f' % (100*corr_values[x]) for x in corr_types])

        return corr_type_to_res
    
    def compute_mutual_correlation(self, metric_to_score_list, metric_to_missing_inds, plot=True):
        all_metrics = self.get_all_metrics(True)
        n = len(all_metrics)
        corr_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                metric1 = all_metrics[i]
                metric2 = all_metrics[j]
                joint_missing_inds = metric_to_missing_inds[metric1].union(metric_to_missing_inds[metric2])
                metric1_score_list = [metric_to_score_list[metric1][i] for i in range(len(metric_to_score_list[metric1])) if i not in joint_missing_inds]
                metric2_score_list = [metric_to_score_list[metric2][i] for i in range(len(metric_to_score_list[metric2])) if i not in joint_missing_inds]
                cur_corr = stats.pearsonr(metric1_score_list, metric2_score_list).statistic
                corr_mat[i, j] = cur_corr
                corr_mat[j, i] = cur_corr

        if plot:
            def shorten_name(metric_name):
                name_parts = metric_name.split()
                if len(name_parts) > 2:
                    metric_name = name_parts[0] + ' ' + ''.join([x.capitalize()[0] for x in name_parts[1:]])
                return metric_name

            all_metrics_for_plot = [shorten_name(x) for x in all_metrics]
            fig, ax = plt.subplots()
            im = ax.imshow(corr_mat)
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel('Correlation', rotation=-90, va="bottom")
            ax.set_xticks(np.arange(n), labels=all_metrics_for_plot)
            ax.set_yticks(np.arange(n), labels=all_metrics_for_plot)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            # for i in range(n):
            #     for j in range(n):
            #         text = ax.text(j, i, '%.2f' % corr_mat[i, j], ha="center", va="center", color="w", fontsize=6)
            # ax.set_title('Mutual correlation between metrics')
            #blue_patch = mpatches.Patch(color='blue', label='Lexical similarity')
            red_patch = mpatches.Patch(color='red', label='Transformer based')
            #plt.legend(handles=[blue_patch, red_patch])
            plt.legend(handles=[red_patch], bbox_to_anchor=(0, -0.2))
            fig.tight_layout()
            plt.savefig('mutual_corr.png')

    def pairwise_comparison(self, ensemble_weights=None, dataset_name=None, tested_type=None):
        all_metrics = self.get_all_metrics()
        if ensemble_weights is not None:
            all_metrics.append('ensemble')
        metric_to_correct_count = {metric: 0 for metric in all_metrics}
        metric_to_all_count = {metric: 0 for metric in all_metrics}
        pair_limit = None
        if dataset_name is None:
            dataset_items = self.data.items()
        else:
            dataset_items = [(dataset_name, self.data[dataset_name])]
        for dataset_name, dataset_data in dataset_items:
            for image_data in tqdm(dataset_data.values(), desc=f'Pairwise comparison on {self.get_name()}, {dataset_name}'):
                if 'pair' in image_data['captions'][0]:
                    pairs_for_comparison = []
                    visited_inds = set()
                    for caption_ind, caption_data in enumerate(image_data['captions']):
                        if caption_ind in visited_inds:
                            continue
                        cur_pair = caption_data['pair']
                        pairs_for_comparison.append((caption_ind, cur_pair))
                        visited_inds.add(caption_ind)
                        visited_inds.add(cur_pair)
                else:
                    n = len(image_data['captions'])
                    pairs_for_comparison = [(i, j) for i in range(n) for j in range(i+1, n)]
                if pair_limit is not None:
                    pairs_for_comparison = random.sample(pairs_for_comparison, pair_limit)
                for cur_pair in pairs_for_comparison:
                    first_caption = image_data['captions'][cur_pair[0]]
                    second_caption = image_data['captions'][cur_pair[1]]
                    if tested_type is not None:
                        system_types = sorted([first_caption['type'], second_caption['type']])
                        if tested_type == 'HC' and system_types != ['H', 'H']:
                            continue
                        if tested_type == 'HI' and system_types != ['H', 'R']:
                            continue
                        if tested_type == 'HM' and system_types != ['H', 'M']:
                            continue
                        if tested_type == 'MM' and system_types != ['M', 'M']:
                            continue
                    first_hr = first_caption['human_ratings'][0] # Assuming a single rating
                    second_hr = second_caption['human_ratings'][0] # Assuming a single rating
                    if first_hr == second_hr:
                        # Since human ratings are discrete in many cases they are far more likely to be equal, while the metrics
                        # are usually continuous. Ignore such cases
                        continue
                    for metric in all_metrics:
                        if metric != 'ensemble' and (metric not in first_caption['automatic_metrics'] or metric not in second_caption['automatic_metrics']):
                            continue
                        metric_to_all_count[metric] += 1
                        if metric == 'ensemble':
                            first_predicted = self.predict_with_ensemble_weights(first_caption['automatic_metrics'], ensemble_weights)
                            second_predicted = self.predict_with_ensemble_weights(second_caption['automatic_metrics'], ensemble_weights)
                        else:
                            first_predicted = first_caption['automatic_metrics'][metric]
                            second_predicted = second_caption['automatic_metrics'][metric]
                        if first_hr > second_hr and first_predicted > second_predicted:
                            metric_to_correct_count[metric] += 1
                        elif first_hr < second_hr and first_predicted < second_predicted:
                            metric_to_correct_count[metric] += 1
        
        metric_to_accuracy = {x[0]: x[1]/metric_to_all_count[x[0]] for x in metric_to_correct_count.items()}
        res = [(metric, metric_to_accuracy[metric]) for metric in all_metrics]
        res.sort(key=lambda x:x[1], reverse=True)
        return res
