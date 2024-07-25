import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os
import sys
import random
import pickle
import torch
import torch.nn as nn
import statistics
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
    
    def get_data_list(self, split_name, img_obj=False):
        image_ids = []
        image_paths = []
        images = []
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
                if img_obj:
                    images.append(self.get_image(split_name, image_data))

        if img_obj:
            return image_ids, images, caption_inds, references, candidates
        else:
            return image_ids, image_paths, caption_inds, references, candidates
    
    def log_scores(self, split_name, image_ids, caption_inds, metric_name, scores):
        for image_id, caption_ind, score in zip(image_ids, caption_inds, scores):
            self.data[split_name][image_id]['captions'][caption_ind]['automatic_metrics'][metric_name] = float(score)

    def compute_metrics(self, compute_clip_image_score=False):
        for split in self.data.keys():
            self.compute_metrics_for_split(split, compute_clip_image_score)

    def compute_metrics_for_split(self, split_name, compute_clip_image_score):
        self.compute_coco_metrics(split_name)
        self.compute_clipscore(split_name)
        self.compute_content_overlap_metrics(split_name)
        self.compute_blip2(split_name)
        self.compute_polos(split_name)
        self.compute_mpnet_score(split_name)
        self.compute_pacscore(split_name)
        if compute_clip_image_score:
            self.compute_clip_image_score(split_name)

    def compute_coco_metrics(self, split_name):
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice
        from collections import OrderedDict

        # Some metrics are not compatabile with large image ids; map to small ones
        new_to_orig_image_id = list(self.data[split_name].keys())
        orig_to_new_image_id = {new_to_orig_image_id[i]: i for i in range(len(new_to_orig_image_id))}
        image_num = len(new_to_orig_image_id)
        digit_num = len(str(image_num))
        orig_to_new_id = lambda image_id, caption_ind: caption_ind*10**(digit_num) + orig_to_new_image_id[image_id]
        new_to_orig_id = lambda new_id: (new_to_orig_image_id[new_id % 10**(digit_num)], new_id // 10**(digit_num))

        # Collect references and candidates
        image_ids, _, caption_inds, references, candidates = self.get_data_list(split_name)
        ref_dict = {}
        cand_dict = {}
        for orig_image_id, caption_ind, refs, cand in zip(image_ids, caption_inds, references, candidates):
            new_id = orig_to_new_id(orig_image_id, caption_ind)
            ref_dict[new_id] = refs
            cand_dict[new_id] = [cand]

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
        image_ids = []
        caption_inds = []
        for id in ref_ids:
            orig_image_id, caption_ind = new_to_orig_id(id)
            image_ids.append(orig_image_id)
            caption_inds.append(caption_ind)
        for metric_name, scores in metric_name_to_scores.items():
            self.log_scores(split_name, image_ids, caption_inds, metric_name, scores)
    
    def compute_clipscore(self, split_name):
        import clip
        sys.path.append('clipscore')
        import clipscore

        device = torch.device('cuda')
        model, transform = clip.load("ViT-B/32", device=device, jit=False)
        model.eval()

        image_ids, image_paths, caption_inds, references, candidates = self.get_data_list(split_name)
        
        image_feats = clipscore.extract_all_images(image_paths, model, device, batch_size=64, num_workers=8)
        _, per_instance_image_text, candidate_feats = clipscore.get_clip_score(model, image_feats, candidates, device)
        _, per_instance_text_text = clipscore.get_refonlyclipscore(model, references, candidate_feats, device)
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

        self.log_scores(split_name, image_ids, caption_inds, 'CLIPScore', per_instance_image_text)
        self.log_scores(split_name, image_ids, caption_inds, 'RefCLIPScore', refclipscores)
    
    def compute_pacscore(self, split_name):
        image_ids, image_paths, caption_inds, references, candidates = self.get_data_list(split_name)

        gen = {}
        gts = {}

        ims_cs = list()
        gen_cs = list()
        gts_cs = list()

        for i, (im_i, gts_i, gen_i) in enumerate(zip(image_paths, references, candidates)):
            gen['%d' % (i)] = [gen_i, ]
            gts['%d' % (i)] = gts_i
            ims_cs.append(im_i)
            gen_cs.append(gen_i)
            gts_cs.append(gts_i)

        sys.path.append('pacscore')
        import evaluation
        gts = evaluation.PTBTokenizer.tokenize(gts)
        gen = evaluation.PTBTokenizer.tokenize(gen)

        from models.clip import clip
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device=device)
        model = model.to(device)
        model = model.float()
        checkpoint = torch.load("pacscore/checkpoints/clip_ViT-B-32.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        from evaluation import PACScore, RefPACScore
        _, pac_scores, candidate_feats, len_candidates = PACScore(model, preprocess, ims_cs, gen_cs, device, w=2.0)
        _, per_instance_text_text = RefPACScore(model, gts_cs, candidate_feats, device, torch.tensor(len_candidates))
        refpac_scores = 2 * pac_scores * per_instance_text_text / (pac_scores + per_instance_text_text)

        self.log_scores(split_name, image_ids, caption_inds, 'PACScore', pac_scores)
        self.log_scores(split_name, image_ids, caption_inds, 'RefPACScore', refpac_scores)
        
        self.clean_temp_files()
    
    def compute_content_overlap_metrics(self, split_name):
        from content_score import compute_and_add_content_recall

        image_ids, _, caption_inds, references, candidates = self.get_data_list(split_name)
        samples = [{'candidate_summary': candidate, 'refs': refs} for candidate, refs in zip(candidates, references)]

        res = compute_and_add_content_recall(samples, 'refs')
        self.log_scores(split_name, image_ids, caption_inds, 'Exact noun overlap', [x['scores']['content_recall']['candidate_summary_noun_recall'] for x in res])
        self.log_scores(split_name, image_ids, caption_inds, 'Fuzzy noun overlap', [x['scores']['content_recall']['candidate_summary_noun_fuzzy_recall'] for x in res])
        self.log_scores(split_name, image_ids, caption_inds, 'Exact verb overlap', [x['scores']['content_recall']['candidate_summary_verb_recall'] for x in res])
        self.log_scores(split_name, image_ids, caption_inds, 'Fuzzy verb overlap', [x['scores']['content_recall']['candidate_summary_verb_fuzzy_recall'] for x in res])
    
    def compute_polos(self, split_name):
        from polos.models import download_model, load_checkpoint

        image_ids, images, caption_inds, references, candidates = self.get_data_list(split_name, img_obj=True)
        polos_data = [{'img': images[i], 'mt': candidates[i], 'refs': references[i]} for i in range(len(images))]

        print('Loading model...', flush=True)
        model_path = download_model("polos")
        model = load_checkpoint(model_path)
        print('Model loaded!')
        print('Computing scores...', flush=True)
        _, scores = model.predict(polos_data, batch_size=8, cuda=True)

        self.log_scores(split_name, image_ids, caption_inds, 'polos', scores)

    def compute_clip_image_score(self, split_name):
        from diffusers import AutoPipelineForText2Image
        import clip

        device = torch.device('cuda')
        pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)

        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        cos_sim = nn.CosineSimilarity()

        # Collect references and candidates
        temp_file = f'{self.get_name()}_{split_name}_tmp.pkl'
        if os.path.isfile(temp_file):
            print(f'Loading data from file: {temp_file}', flush=True)
            with open(temp_file, 'rb') as fp:
                self.data = pickle.load(fp)
        count = 0
        image_ids, images, caption_inds, _, candidates = self.get_data_list(split_name, img_obj=True)
        prev_image_id = None
        scores = []
        with torch.no_grad():
            for image_id, orig_image, caption_ind, candidate in tqdm(zip(image_ids, images, caption_inds, candidates)):
                if image_id != prev_image_id:
                    if count % 100 == 0:
                        with open(temp_file, 'wb') as fp:
                            pickle.dump(self.data, fp)
                    count += 1
                    orig_image = preprocess(orig_image).unsqueeze(0).to(device)
                    orig_image_features = clip_model.encode_image(orig_image)
                if 'CLIPImageScore' in self.data[split_name][image_id]['captions'][caption_ind]['automatic_metrics']:
                    continue
                reconstructed_image = pipeline_text2image(prompt=candidate).images[0]
                reconstructed_image = preprocess(reconstructed_image).unsqueeze(0).to(device)
                reconstructed_image_features = clip_model.encode_image(reconstructed_image)
                score = cos_sim(orig_image_features, reconstructed_image_features).item()
                scores.append(score)

        self.log_scores(split_name, image_ids, caption_inds, 'CLIPImageScore', scores)

    def numpy_cosine_similarity(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    def compute_mpnet_score(self, split_name, agg_method='mean'):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-mpnet-base-v2')
        model.eval()

        image_ids, _, caption_inds, references, candidates = self.get_data_list(split_name)
        scores = []
        for candidate, refs in zip(candidates, references):
            with torch.no_grad():
                cand_embedding = model.encode(candidate)
                ref_embeddings = [model.encode(ref) for ref in refs]
            cur_scores = [self.numpy_cosine_similarity(cand_embedding, ref_embedding).item() for ref_embedding in ref_embeddings]
            if agg_method == 'mean':
                score = statistics.mean(cur_scores)
            elif agg_method == 'max':
                score = max(cur_scores)
            scores.append(score)
        
        self.log_scores(split_name, image_ids, caption_inds, 'MPNetScore', scores)

    def compute_blip2(self, split_name):
        from lavis.models import load_model_and_preprocess
        import imageio.v2 as imageio

        device = torch.device('cuda')
        model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

        image_ids, images, caption_inds, _, candidates = self.get_data_list(split_name, img_obj=True)
        prev_image_id = None
        scores = []
        with torch.no_grad():
            for image_id, raw_image, caption in zip(image_ids, images, candidates):
                if image_id != prev_image_id:
                    # Do this check somehow
                    # im = imageio.imread(file_path)
                    # if len(im.shape) != 3 or im.shape[2] != 3:
                    #     continue
                    img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                    prev_image_id = image_id
                txt = text_processors["eval"](caption)
                score = model({"image": img, "text_input": txt}, match_head='itc')
                scores.append(score)

        self.log_scores(split_name, image_ids, caption_inds, 'BLIP2Score', scores)

    def get_all_metrics(self, sort_by_type=False):
        # all_metrics = list(set([x for dataset_data in self.data.values() for image_data in dataset_data.values() for caption_data in image_data['captions'] for x in caption_data['automatic_metrics'].keys()]))
        # all_metrics = [x for x in all_metrics if not x.startswith('SPICE_')]
        if sort_by_type:
            all_metrics = ['BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'CIDEr', 'Exact noun overlap', 'Exact verb overlap', 'METEOR', 'ROUGE', 'Fuzzy noun overlap', 'Fuzzy verb overlap', 'SPICE', 'MPNetScore', 'BLIP2Score', 'CLIPImageScore', 'CLIPScore', 'PACScore', 'RefCLIPScore', 'RefPACScore', 'polos']
        else:
            all_metrics = ['Exact noun overlap', 'Fuzzy noun overlap', 'Exact verb overlap', 'Fuzzy verb overlap', 'CLIPScore', 'RefCLIPScore', 'METEOR', 'PACScore', 'ROUGE', 'RefPACScore', 'SPICE', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'BLIP2Score', 'CIDEr', 'CLIPImageScore', 'MPNetScore', 'polos']
            all_metrics.sort()

        return all_metrics

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
