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
import nltk.translate.nist_score as nist_score
import os
import json
import sys
import subprocess
import shutil
import pathlib
import pickle
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

dump_file = 'dataset.pkl'

class HumanRatingDataset:
    def __init__(self):
        self.data = {}

    def dump(self):
        with open(dump_file, 'wb') as fp:
            pickle.dump(self.data, fp)

    def load(self):
        with open(dump_file, 'rb') as fp:
            self.data = pickle.load(fp)
    
    def collect_data(self):
        return

    def compute_metrics(self):
        for dataset in self.data.keys():
            self.compute_metrics_for_dataset(dataset)

    def compute_metrics_for_dataset(self, dataset_name):
        self.compute_coco_metrics(dataset_name)
        self.compute_huggingface_metrics(dataset_name)
        self.compute_sentence_level_huggingface_metrics(dataset_name)
        self.compute_sentence_level_nltk_metrics(dataset_name)
        self.compute_clipscore(dataset_name)

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

        # Log scores
        for metric_name, scores in metric_name_to_scores.items():
            for id, score in zip(ref_ids, scores):
                orig_image_id, caption_id = new_to_orig_id(id)
                self.data[dataset_name][orig_image_id]['captions'][caption_id]['automatic_metrics'][metric_name] = score

    def compute_huggingface_metrics(self, dataset_name):
        # Collect references and candidates
        references = []
        candidates = []
        image_id_caption_ind_pairs = []
        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                image_id_caption_ind_pairs.append((image_id, caption_ind))
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                references.append([image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs])
                candidates.append(caption_data['caption'])

        # Now, compute metrics
        metric_name_to_scores = {}

        ####BERTScore###
        bertscore = load("bertscore")
        results = bertscore.compute(predictions=candidates, references=references, lang='en')
        metric_name_to_scores['BERTScore'] = results['f1']

        # Log scores
        for metric_name, scores in metric_name_to_scores.items():
            for sample_info, score in zip(image_id_caption_ind_pairs, scores):
                image_id, caption_id = sample_info
                self.data[dataset_name][image_id]['captions'][caption_id]['automatic_metrics'][metric_name] = score

    def compute_sentence_level_nltk_metrics(self, dataset_name):
        # NIST
        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                candidate = caption_data['caption']
                self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['NIST'] = nist_score.sentence_nist(hypothesis=candidate, references = references)
    
    def get_candidate_num_per_image(self, dataset_name):
        return
    
    def get_file_name2iid_func(self, dataset_name):
        return
    
    def compute_clipscore(self, dataset_name):
        # The CLIPScore metrics require a mapping from image id to candidate. Since we have multiple candidates per image, we need to run it multiple times
        N = self.get_candidate_num_per_image(dataset_name)
        for caption_ind in range(N):
            # First, create a temporary json file with image file names and caption, to be used by clip score
            temp_cands_file_name = f'temp_cands_{dataset_name}.json'
            temp_refs_file_name = f'temp_refs_{dataset_name}.json'
            temp_res_file = f'temp_clipscore.json'
            temp_image_dir = None

            references = {}
            candidates = {}
            image_dir = None
            for image_data in self.data[dataset_name].values():
                caption_data = image_data['captions'][caption_ind]
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                cur_references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                cur_candidate = caption_data['caption']
                file_name = image_data['file_path'].split('/')[-1].split('.')[0]
                references[file_name] = cur_references
                candidates[file_name] = cur_candidate
                cur_image_dir = '/'.join(image_data['file_path'].split('/')[:-1])
                if image_dir is None:
                    image_dir = cur_image_dir
                else:
                    assert cur_image_dir == image_dir, f'Can\'t run clipscore, found images from two different directories:\n{image_dir}\n{cur_image_dir}'

            # CLIPScore expects all the images in the target directory to have a candidate; To make sure this is true, move images to a new directory
            image_paths = [os.path.join(image_dir, path) for path in os.listdir(image_dir) if path.endswith(('.png', '.jpg', '.jpeg', '.tiff'))]
            image_ids = [pathlib.Path(path).stem for path in image_paths]
            if len(image_ids) > len([x for x in image_ids if x in candidates]):
                temp_image_dir = f'temp_{dataset_name}_images'
                os.mkdir(temp_image_dir)
                for file_name in candidates.keys():
                    _ = shutil.copy(os.path.join(image_dir, f'{file_name}.jpg'), temp_image_dir)
                image_dir = temp_image_dir

            with open(temp_cands_file_name, 'w') as fp:
                fp.write(json.dumps(candidates))

            with open(temp_refs_file_name, 'w') as fp:
                fp.write(json.dumps(references))

            _ = subprocess.call([sys.executable, 'clipscore/clipscore.py',
                                 temp_cands_file_name,
                                 image_dir,
                                 '--references_json', temp_refs_file_name,
                                 '--compute_other_ref_metrics', '0',
                                 '--save_per_instance', temp_res_file])
            
            # Log results
            with open(temp_res_file, 'r') as fp:
                results = json.load(fp)

            file_name2iid = self.get_file_name2iid_func(dataset_name)
            for file_name, score_dict in results.items():
                image_id = file_name2iid(file_name)
                for metric, score in score_dict.items():
                    self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics'][metric] = score

            # Now, delete the temporary files
            os.remove(temp_cands_file_name)
            os.remove(temp_refs_file_name)
            os.remove(temp_res_file)
            if temp_image_dir is not None:
                shutil.rmtree(temp_image_dir)
    
    def compute_sentence_level_huggingface_metrics(self, dataset_name):
        ter = load('ter')

        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                candidate = caption_data['caption']
                self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['TER'] = ter.compute(predictions=[candidate], references=[references])['score']

    def get_all_metrics(self):
        all_metrics = list(set([x for dataset_data in self.data.values() for image_data in dataset_data.values() for caption_data in image_data['captions'] for x in caption_data['automatic_metrics'].keys()]))
        all_metrics = [x for x in all_metrics if not x.startswith('SPICE_')]
        all_metrics.sort()

        return all_metrics

    def compute_correlation(self):
        all_metrics = self.get_all_metrics()
        human_rating_list = []
        metric_to_score_list = {metric: [] for metric in all_metrics}
        metric_to_missing_inds = {metric: set() for metric in all_metrics}
        for dataset_data in self.data.values():
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
        all_metrics = self.get_all_metrics()
        
        metric_to_corr = {}
        for metric in all_metrics:
            cur_human_rating_list = [human_rating_list[i] for i in range(len(human_rating_list)) if i not in metric_to_missing_inds[metric]]
            cur_metric_score_list = [metric_to_score_list[metric][i] for i in range(len(metric_to_score_list[metric])) if i not in metric_to_missing_inds[metric]]
            metric_to_corr[metric] = stats.pearsonr(cur_human_rating_list, cur_metric_score_list)

        res = [(metric, metric_to_corr[metric].statistic) for metric in all_metrics]
        res.sort(key=lambda x:x[1], reverse=True)
        return res
    
    def compute_mutual_correlation(self, metric_to_score_list, metric_to_missing_inds):
        all_metrics = self.get_all_metrics()
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

    def select_predictor_metrics(self):
        all_metrics = self.get_all_metrics()
        all_metrics.sort()

        N = sum([sum([len(image_data['captions']) for image_data in dataset_data.values()]) for dataset_data in self.data.values()])
        X = np.zeros((N, len(all_metrics)))
        y = np.zeros(N)

        cur_sample_ind = 0
        for dataset_data in self.data.values():
            for image_data in dataset_data.values():
                for caption_data in image_data['captions']:
                    y[cur_sample_ind] = caption_data['human_rating']
                    for metric_ind, metric in enumerate(all_metrics):
                        X[cur_sample_ind, metric_ind] = caption_data['automatic_metrics'][metric]
                    cur_sample_ind += 1

        reg = LinearRegression()
        res = {}
        for direction in ['forward', 'backward']
            sfs = SequentialFeatureSelector(reg, direction=direction)
            sfs.fit(X, y)
            res[direction] = [all_metrics[i] for i in range(len(all_metrics)) if sfs.get_support()[i]]

        return res
