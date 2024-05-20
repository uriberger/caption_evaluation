from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from evaluate import load
from collections import OrderedDict
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

class HumanRatingDataset:
    def __init__(self):
        self.data = {}
    
    def collect_data(self):
        return

    def compute_metrics(self):
        for dataset in self.data.keys():
            self.compute_metrics_for_dataset(dataset)

    def compute_metrics_for_dataset(self, dataset_name):
        self.compute_coco_metrics(dataset_name)

    def compute_coco_metrics(self, dataset_name):
        # Some metrics are not compatabile with large image ids; map to small ones
        new_to_orig_image_id = list(self.data[dataset_name].keys())
        orig_to_new_image_id = {new_to_orig_image_id[i]: i for i in range(len(new_to_orig_image_id))}
        image_num = len(new_to_orig_image_id)
        digit_num = len(str(image_num))
        orig_to_new_id = lambda image_id, caption_ind: caption_ind*10**(digit_num) + orig_to_new_image_id[image_id]
        new_to_orig_id = lambda new_id: (new_to_orig_image_id[new_id % 10**(digit_num)], new_id // 10**(digit_num))

        # Collect references and candidates
        references = {}
        candidates = {}
        for orig_image_id in new_to_orig_image_id:
            image_data = self.data[dataset_name][orig_image_id]
            for caption_ind, caption_data in enumerate(image_data['captions']):
                new_id = orig_to_new_id(orig_image_id, caption_ind)
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                references[new_id] = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                candidates[new_id] = [caption_data['caption']]

        # Tokenize
        tokenizer = PTBTokenizer()
        tokenized_references = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in references.items()})
        tokenized_candidates = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in candidates.items()})

        # Put refs and cands in ordered dict; this is needed so that the results of all metrics are in the image id order
        ref_ids = sorted(list(tokenized_references.keys()))
        assert set(ref_ids) == set(tokenized_candidates.keys())
        ordered_tokenized_references = OrderedDict()
        ordered_tokenized_candidates = OrderedDict()
        for id in ref_ids:
            ordered_tokenized_references[id] = tokenized_references[id]
            ordered_tokenized_candidates[id] = tokenized_candidates[id]
        tokenized_references = ordered_tokenized_references
        tokenized_candidates = ordered_tokenized_candidates

        # Now, compute metrics
        metric_name_to_scores = {}

        ####BLEU####
        pycoco_bleu = Bleu()
        _, all_scores = pycoco_bleu.compute_score(tokenized_references, tokenized_candidates)
        for i in range(4):
            metric_name_to_scores[f'BLEU{i+1}'] = all_scores[i]

        ####METEOR###
        pycoco_meteor = Meteor()
        _, all_scores = pycoco_meteor.compute_score(tokenized_references, tokenized_candidates)
        metric_name_to_scores['METEOR'] = all_scores
        del pycoco_meteor

        ####ROUGE###
        pycoco_rouge = Rouge()
        _, all_scores = pycoco_rouge.compute_score(tokenized_references, tokenized_candidates)
        metric_name_to_scores['ROUGE'] = all_scores

        ####CIDER###
        pycoco_cider = Cider()
        _, all_scores = pycoco_cider.compute_score(tokenized_references, tokenized_candidates)
        metric_name_to_scores['CIDEr'] = all_scores

        ####SPICE###
        pycoco_spice = Spice()
        _, spice_scores = pycoco_spice.compute_score(tokenized_references, tokenized_candidates)
        metric_name_to_scores['SPICE'] = [x['All']['f'] for x in spice_scores]
        spice_submetrics = ['Relation', 'Cardinality', 'Attribute', 'Size', 'Color', 'Object']
        for submetric in spice_submetrics:
            metric_name_to_scores[f'SPICE_{submetric}'] = [x[submetric]['f'] for x in spice_scores]        

        ####BERTScore###
        # bertscore = load("bertscore")
        # reference_list = list(references.items())
        # reference_list.sort(key=lambda x:x[0])
        # references = [x[1] for x in reference_list]
        # prediction_list = list(candidates.items())
        # prediction_list.sort(key=lambda x:x[0])
        # predictions = [x[1][0] for x in prediction_list]
        # results = bertscore.compute(predictions=predictions, references=references, lang='en')
        # bertscore = statistics.mean(results['f1'])

        # Log scores
        for id in ref_ids:
            orig_image_id, caption_id = new_to_orig_id(id)
            self.data[dataset_name][orig_image_id]['captions'][caption_id]['automatic_metrics'] = {}
        
        for metric_name, scores in metric_name_to_scores.items():
            for id, score in zip(ref_ids, scores):
                orig_image_id, caption_id = new_to_orig_id(id)
                self.data[dataset_name][orig_image_id]['captions'][caption_id]['automatic_metrics'][metric_name] = score

    def compute_correlation(self):
        all_metrics = list(set([x for dataset_data in self.data.values() for image_data in dataset_data.values() for caption_data in image_data['captions'] for x in caption_data['automatic_metrics'].keys()]))
        human_rating_list = []
        metric_to_score_list = {metric: [] for metric in all_metrics}
        metric_to_missing_inds = {metric: set() for metric in all_metrics}
        for dataset_data in data.values():
            for image_data in dataset_data.values():
                for caption_data in image_data['captions']:
                    for metric in all_metrics:
                        if metric not in caption_data['automatic_metrics'] or np.isnan(caption_data['automatic_metrics'][metric]):
                            metric_to_missing_inds[metric].add(len(human_rating_list))
                            metric_to_score_list[metric].append(np.nan)
                        else:
                            metric_to_score_list[metric].append(caption_data['automatic_metrics'][metric])
                    human_rating_list.append(caption_data['human_rating'])

        self.compute_mutual_correlation(metric_to_score_list, metric_to_missing_inds)
        return self.compute_correlation_with_human_ratings(human_rating_list, metric_to_score_list, metric_to_missing_inds)
    
    def compute_correlation_with_human_ratings(self, human_rating_list, metric_to_score_list, metric_to_missing_inds):
        all_metrics = list(metric_to_score_list.keys())
        
        metric_to_corr = {}
        for metric in all_metrics:
            cur_human_rating_list = [human_rating_list[i] for i in range(len(human_rating_list)) if i not in metric_to_missing_inds[metric]]
            cur_metric_score_list = [metric_to_score_list[metric][i] for i in range(len(metric_to_score_list[metric])) if i not in metric_to_missing_inds[metric]]
            metric_to_corr[metric] = stats.pearsonr(cur_human_rating_list, cur_metric_score_list)

        res = [(metric, metric_to_corr[metric].statistic) for metric in all_metrics]
        res.sort(key=lambda x:x[1], reverse=True)
        return res
    
    def compute_mutual_correlation(self, metric_to_score_list, metric_to_missing_inds):
        all_metrics = sorted(list(metric_to_score_list.keys()))
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

        fig, ax = plt.subplots()
        im = ax.imshow(corr_mat)
        ax.set_xticks(np.arange(n), labels=all_metrics)
        ax.set_yticks(np.arange(n), labels=all_metrics)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        for i in range(n):
            for j in range(n):
                text = ax.text(j, i, '%.2f' % corr_mat[i, j], ha="center", va="center", color="w", fontsize=6)
        ax.set_title('Mutual correlation between metrics')
        fig.tight_layout()
        plt.savefig('mutual_corr.png')
