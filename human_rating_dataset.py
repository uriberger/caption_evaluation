from collections import OrderedDict
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
sys.path.append('NNEval')
import subprocess
import shutil
import pathlib
import pickle
import math
import torch
import torch.nn as nn
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
import statistics
from tqdm import tqdm
from PIL import Image

class_to_pos_tag = [
    # Nouns:
    ['NN', 'NNS', 'NNP', 'WP', 'NNPS', 'WP$'],
    # Verbs:
    ['VBD', 'VB', 'VBP', 'VBG', 'VBZ', 'VBN', 'VERB'],
    # Adjectivs:
    ['JJ', 'JJR', 'JJS'],
    # Others:
    ['<unk>', 'UH', ',', 'PRP', 'PRP$', 'RB', '.', 'DT', 'O', 'IN', 'CD', 'WRB', 'WDT',
     'CC', 'TO', 'MD', ':', 'RP', 'EX', 'FW', 'XX', 'HYPH', 'POS', 'RBR', 'PDT', 'RBS',
     'AFX', '-LRB-', '-RRB-', '``', "''", 'LS', '$', 'SYM', 'ADD', '*', 'NFP']
]
pos_tag_to_class = {}
for class_ind in range(len(class_to_pos_tag)):
    for pos_tag in class_to_pos_tag[class_ind]:
        pos_tag_to_class[pos_tag] = class_ind

class HumanRatingDataset:
    def __init__(self):
        self.data = {}

    def get_name(self):
        return ''

    def dump(self):
        dump_file = f'{self.get_name()}_data.pkl'
        with open(dump_file, 'wb') as fp:
            pickle.dump(self.data, fp)

    def load(self):
        dump_file = f'{self.get_name()}_data.pkl'
        with open(dump_file, 'rb') as fp:
            self.data = pickle.load(fp)
    
    def collect_data(self):
        return
    
    def get_image(self, dataset_name, image_data):
        return
    
    def get_file_path(self, dataset_name, image_data):
        return
    
    def clean_temp_files(self):
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
        self.compute_smurf(dataset_name)
        self.compute_wmd(dataset_name)
        self.compute_content_overlap_metrics(dataset_name)

    def compute_coco_metrics(self, dataset_name):
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.spice.spice import Spice

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
        from evaluate import load

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
        import nltk.translate.nist_score as nist_score

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
        return max([len(image_data['captions']) for image_data in self.data[dataset_name].values()])
    
    def get_file_name2iid_func(self, dataset_name):
        return
    
    def compute_clipscore(self, dataset_name):
        # The CLIPScore metrics require a mapping from image id to candidate. Since we have multiple candidates per image, we need to run it multiple times
        N = self.get_candidate_num_per_image(dataset_name)
        for caption_ind in tqdm(range(N)):
            # First, create a temporary json file with image file names and caption, to be used by clip score
            temp_cands_file_name = f'temp_cands_{dataset_name}.json'
            temp_refs_file_name = f'temp_refs_{dataset_name}.json'
            temp_res_file = f'temp_clipscore.json'
            temp_image_dir = None

            references = {}
            candidates = {}
            image_dir = None
            for image_data in self.data[dataset_name].values():
                if caption_ind >= len(image_data['captions']):
                    continue
                caption_data = image_data['captions'][caption_ind]
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                cur_references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                cur_candidate = caption_data['caption']
                file_path = self.get_file_path(dataset_name, image_data)
                file_name = file_path.split('/')[-1].split('.')[0]
                cur_image_dir = '/'.join(file_path.split('/')[:-1])
                references[file_name] = cur_references
                candidates[file_name] = cur_candidate
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
            self.clean_temp_files()
    
    def compute_sentence_level_huggingface_metrics(self, dataset_name):
        ter = load('ter')

        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                candidate = caption_data['caption']
                self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['TER'] = (-1)*ter.compute(predictions=[candidate], references=[references])['score']

    def compute_smurf(self, dataset_name):
        from SMURF.smurf.eval import smurf_eval_captions

        image_id_caption_ind_pairs = []
        references = []
        candidates = []
        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                cur_refs = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                cur_cand = caption_data['caption']
                references.append(cur_refs)
                candidates.append(cur_cand)
                image_id_caption_ind_pairs.append((image_id, caption_ind))

        meta_scorer = smurf_eval_captions(references, candidates, fuse=True)
        os.chdir('SMURF')
        scores = meta_scorer.evaluate()
        os.chdir('..')
        for sample_entry, score in zip(image_id_caption_ind_pairs, scores['SMURF']):
            image_id, caption_ind = sample_entry
            self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['SMURF'] = score
    
    def compute_wmd(self, dataset_name, agg_method='mean'):
        import gensim.downloader as api
        from nltk.corpus import stopwords
        from nltk import download

        model = api.load('word2vec-google-news-300')
        _ = download('stopwords')
        stop_words = stopwords.words('english')

        def preprocess(sentence):
            return [w for w in sentence.lower().split() if w not in stop_words]
        
        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                candidate = caption_data['caption']

                references = [preprocess(x) for x in references]
                candidate = preprocess(candidate)
                similarities = [(-1)*model.wmdistance(candidate, x) for x in references]
                if agg_method == 'mean':
                    similarity = statistics.mean(similarities)
                elif agg_method == 'max':
                    similarity = max(similarities)
                if similarity == -math.inf:
                    similarity = 0
                self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['WMD'] = similarity
    
    def compute_nubia(self, dataset_name, agg_method):
        generated_scores_file_name = f'{dataset_name}_nubia_scores.pkl'
        assert os.path.isfile(generated_scores_file_name)
        with open(generated_scores_file_name, 'rb') as fp:
            data = pickle.load(fp)
        for image_id, image_scores in data.items():
            for caption_ind, scores in enumerate(image_scores):
                if agg_method == 'mean':
                    score = statistics.mean(scores)
                elif agg_method == 'max':
                    score = max(scores)
                self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['nubia'] = score

    def compute_umic(self, dataset_name):
        # The umic metrics require a mapping from image id to candidate. Since we have multiple candidates per image, we need to run it multiple times
        N = self.get_candidate_num_per_image(dataset_name)
        for caption_ind in range(N):
            # First, create a temporary json file with image file names and caption, to be used by umic
            temp_cands_file_name = f'temp_cands_{dataset_name}.json'
            temp_res_file = f'temp_umic.json'
            temp_txt_db_dir = f'temp_txt_db'

            candidates = []
            for image_data in self.data[dataset_name].values():
                caption_data = image_data['captions'][caption_ind]
                cur_candidate = caption_data['caption']
                file_path = self.get_file_path(dataset_name, image_data)
                file_name = file_path.split('/')[-1].split('.')[0]
                candidates.append({'caption': cur_candidate, 'imgid': file_name})

            with open(temp_cands_file_name, 'w') as fp:
                fp.write(json.dumps(candidates))

            if dataset_name.startswith('flickr'):
                img_type = dataset_name
            elif dataset_name.startswith('coco'):
                img_type = 'coco_val2014'
            else:
                assert False
            _ = subprocess.call(['UMIC/venv/bin/python', 'UMIC/make_txt_db.py',
                                 '--input_file', temp_cands_file_name,
                                 '--img_type', img_type,
                                 '--out_dir', temp_txt_db_dir])
            
            _ = subprocess.call(['UMIC/venv/bin/python', 'UMIC/compute_metric.py',
                                 '--img_db', img_type,
                                 '--txt_db', temp_txt_db_dir,
                                 '--out_file', temp_res_file])
            
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
            os.remove(temp_res_file)
            shutil.rmtree(temp_txt_db_dir)
            self.clean_temp_files()
    
    def compute_nneval(self, dataset_name):
        import gensim
        import NNEval.configuration_nneval as configuration
        from NNEval.nn_classify_model_nneval import build_model
        import tensorflow as tf

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
        ref = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in references.items()})
        hypo = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in candidates.items()})
    
        assert(hypo.keys() == ref.keys())
        ImgId=hypo.keys() # for ensuring that all metrics get the same keys and return values in the same order
        stop_words = stopwords.words('english')
        embeddings = gensim.models.KeyedVectors.load_word2vec_format( "NNEval/GoogleNews-vectors-negative300.bin" , binary=True ) 

        _, blscores=Bleu(4).compute_score(ref,hypo)
        #Rogue_L,Rogue_Lscores= Rouge().compute_score(ref,hypo,ImgId)
        _, meteor_scores = Meteor().compute_score(ref,hypo)
        
        wmd_model = embeddings
        wmd_model.init_sims(replace=True)
        wmd_score=[]
        for id_ref in ImgId:
            c1=hypo[id_ref][0]
            c1= c1.lower().split()
            c1 = [w_ for w_ in c1 if w_ not in stop_words]    
            distance=[]
            for refs in ref[id_ref]:
                c2=refs
                c2= c2.lower().split()
                c2 = [w_ for w_ in c2 if w_ not in stop_words]
                temp= wmd_model.wmdistance(c1, c2)
                if (np.isinf(temp)):
                    temp=1000
                distance.append(temp)
            wmd_dis=min(distance)
            wmd_similarity=np.exp(-wmd_dis)
            wmd_score.append(wmd_similarity)
    
        CIDer=Cider()
        _, cider_scores=CIDer.compute_score(ref,hypo)

        _, spice_scores = Spice().compute_score(ref,hypo)

        features={}
        for i,ids in enumerate(ImgId):     
            features[ids]=[
                            meteor_scores[i],
                            wmd_score[i],
                            blscores[0][i],
                            blscores[1][i],
                            blscores[2][i],
                            blscores[3][i],
                            spice_scores[i],
                            cider_scores[i]/10]
            
        model_config = configuration.ModelConfig()

        def _step_test(sess, model, features):
            nn_score= sess.run([model['nn_score']],  feed_dict={model['sentence_features']: features})            
            return nn_score
        
        def process_scores(sc,BATCH_SIZE_INFERENCE):
            score_list=[]
            for i in range(BATCH_SIZE_INFERENCE):
                score_list.append(sc[0][i][1])
            return score_list

        g = tf.Graph()
        with g.as_default():
            model = build_model(model_config)
            init = tf.global_variables_initializer()
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
            checking=[1110] # model number
            with tf.device('/gpu:1'): 
                sess.run(init)
                for s in checking:
                    model['saver'].restore(sess, os.path.join(directory,'nn_classify_checkpoint{}.ckpt'.format(s)))
                    for i in range(1):
                        sc = _step_test(sess,model,features) 
                    scores=process_scores(sc, len(features))
                sess.close()

    def compute_pacscore(self, dataset_name):
        image_ids = []
        image_paths = []
        caption_inds = []
        references = []
        candidates = []
        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                cur_refs = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                cur_cand = caption_data['caption']
                references.append(cur_refs)
                candidates.append(cur_cand)
                file_path = self.get_file_path(dataset_name, image_data)
                image_paths.append(file_path)
                image_ids.append(image_id)
                caption_inds.append(caption_ind)

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

        for image_id, caption_ind, pac_score, refpac_score in zip(image_ids, caption_inds, pac_scores, refpac_scores):
            self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['PAC'] = pac_score
            self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['RefPAC'] = refpac_score

        self.clean_temp_files()
    
    def compute_content_overlap_metrics(self, dataset_name):
        from content_score import compute_and_add_content_recall

        # Collect references and candidates
        samples = []
        image_id_caption_ind_pairs = []
        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                image_id_caption_ind_pairs.append((image_id, caption_ind))
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                candidate = caption_data['caption']
                references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                samples.append({'candidate_summary': candidate, 'refs': references})

        res = compute_and_add_content_recall(samples, 'refs')

        for sample_info, cur_res in zip(image_id_caption_ind_pairs, res):
            image_id, caption_id = sample_info
            self.data[dataset_name][image_id]['captions'][caption_id]['automatic_metrics']['Exact noun overlap'] = cur_res['scores']['content_recall']['candidate_summary_noun_recall']
            self.data[dataset_name][image_id]['captions'][caption_id]['automatic_metrics']['Fuzzy noun overlap'] = cur_res['scores']['content_recall']['candidate_summary_noun_fuzzy_recall']
            self.data[dataset_name][image_id]['captions'][caption_id]['automatic_metrics']['Exact verb overlap'] = cur_res['scores']['content_recall']['candidate_summary_verb_recall']
            self.data[dataset_name][image_id]['captions'][caption_id]['automatic_metrics']['Fuzzy verb overlap'] = cur_res['scores']['content_recall']['candidate_summary_verb_fuzzy_recall']
    
    def compute_polos(self, dataset_name):
        from polos.models import download_model, load_checkpoint

        polos_data = []
        image_id_caption_ind_pairs = []
        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                image_id_caption_ind_pairs.append((image_id, caption_ind))
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                polos_data.append({
                    'img': self.get_image(image_data),
                    'mt': caption_data['caption'],
                    'refs': [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                    })

        print('Loading model...', flush=True)
        model_path = download_model("polos")
        model = load_checkpoint(model_path)
        print('Model loaded!')
        print('Computing scores...', flush=True)
        _, scores = model.predict(polos_data, batch_size=8, cuda=True)

        # Log scores
        for sample_info, score in zip(image_id_caption_ind_pairs, scores):
            image_id, caption_id = sample_info
            self.data[dataset_name][image_id]['captions'][caption_id]['automatic_metrics']['polos'] = score

    def compute_clip_image_score(self, dataset_name):
        from diffusers import AutoPipelineForText2Image
        import clip
        from PIL import Image

        device = torch.device('cuda')
        pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
        ).to(device)

        clip_model, preprocess = clip.load("ViT-B/32", device=device)
        cos_sim = nn.CosineSimilarity()

        # Collect references and candidates
        temp_file = 'tmp.pkl'
        count = 0
        with torch.no_grad():
            for image_id, image_data in tqdm(self.data[dataset_name].items()):
                if count % 100 == 0:
                    with open(temp_file, 'wb') as fp:
                        pickle.dump(self.data, fp)
                count += 1
                orig_image = self.get_image(image_data)
                orig_image = preprocess(orig_image).unsqueeze(0).to(device)
                orig_image_features = clip_model.encode_image(orig_image)
                for caption_ind, caption_data in enumerate(image_data['captions']):
                    candidate = caption_data['caption']
                    reconstructed_image = pipeline_text2image(prompt=candidate).images[0]
                    reconstructed_image = preprocess(reconstructed_image).unsqueeze(0).to(device)
                    reconstructed_image_features = clip_model.encode_image(reconstructed_image)
                    score = cos_sim(orig_image_features, reconstructed_image_features).item()
                    self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['CLIPImageScore'] = score

    def numpy_cosine_similarity(self, a, b):
        return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    
    def compute_mpnet_score(self, dataset_name, agg_method='mean'):
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer('all-mpnet-base-v2')
        model.eval()

        for image_id, image_data in self.data[dataset_name].items():
            for caption_ind, caption_data in enumerate(image_data['captions']):
                ignore_refs = []
                if 'ignore_refs' in caption_data:
                    ignore_refs = caption_data['ignore_refs']
                candidate = caption_data['caption']
                references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
                with torch.no_grad():
                    cand_embedding = model.encode(candidate)
                    ref_embeddings = [model.encode(ref) for ref in references]
                scores = [self.numpy_cosine_similarity(cand_embedding, ref_embedding).item() for ref_embedding in ref_embeddings]
                if agg_method == 'mean':
                    score = statistics.mean(scores)
                elif agg_method == 'max':
                    score = max(scores)
                self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['MPNet'] = score

    def compute_blip2(self, dataset_name):
        from lavis.models import load_model_and_preprocess

        device = torch.device('cuda')
        model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

        for image_id, image_data in tqdm(self.data[dataset_name].items()):
            with torch.no_grad():
                raw_image = self.get_image(image_data)
                img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                for caption_ind, caption_data in enumerate(image_data['captions']):
                    caption = caption_data['caption']
                    txt = text_processors["eval"](caption)
                    score = model({"image": img, "text_input": txt}, match_head='itc')
                    self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['BLIP2Score'] = score.item()

    def compute_retrieval_score(self, dataset_name):
        system_num = self.get_candidate_num_per_image(dataset_name)
        assert len([x for x in image_data.values() if len([]) != system_num]) == 0
        import clip
        from PIL import Image

        device = torch.device('cuda')
        clip_model, preprocess = clip.load("ViT-B/32", device=device)

        image_embeds = {}
        text_embeds = {}
        data_back_ref = []

        # Collect references and candidates
        with torch.no_grad():
            for image_id, image_data in tqdm(self.data[dataset_name].items()):
                image = self.get_image(image_data)
                image = preprocess(image).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                image_embed_ind = len(image_embeds)
                image_embeds[image_embed_ind] = (image_features, [])
                for caption_ind, caption_data in enumerate(image_data['captions']):
                    candidate = caption_data['caption']
                    text = clip.tokenize(candidate).to(device)
                    text_features = clip_model.encode_text(text)
                    text_embed_ind = len(text_embeds)
                    text_embeds[text_embed_ind] = (text_features, image_embed_ind)
                    image_embeds[image_embed_ind][1].append(text_embed_ind)
                    data_back_ref.append((image_id, caption_ind, image_embed_ind, text_embed_ind))

        image_embeds_list = sorted(list(image_embeds.items()), key=lambda x:x[0])
        image_embeds_only = [x[1][0] for x in image_embeds_list]
        image_mat = torch.cat(image_embeds_only)
        text_embeds_list = sorted(list(text_embeds.items()), key=lambda x:x[0])
        text_embeds_only = [x[1][0] for x in text_embeds_list]
        text_mat = torch.cat(text_embeds_only)
        
        sim_mat = image_mat.matmul(text_mat.transpose(1, 0))
        image_sim_sorted = torch.sort(sim_mat, dim=1).values
        text_sim_sorted = torch.sort(sim_mat, dim=0).values
        system_to_correct = {i: {'image_r@1': 0, 'image_r@5': 0, 'text_r@10': 0, 'text_r@1': 0, 'text_r@5': 0, 'text_r@10': 0} for i in range(system_num)}
        system_to_count = {i: {'image_r@1': 0, 'image_r@5': 0, 'text_r@10': 0, 'text_r@1': 0, 'text_r@5': 0, 'text_r@10': 0} for i in range(system_num)}
        # Need to think how to implement this

    def compute_mplug_score(self, dataset_name):
        sys.path.append('mPLUG')
        from mPLUG.models.model_retrieval_mplug import MPLUG
        from mPLUG.models.tokenization_bert import BertTokenizer
        from mPLUG.models.vit import resize_pos_embed
        from ruamel.yaml import YAML
        from torchvision import transforms
        import torch.nn.functional as F
        import torch.nn as nn

        device = torch.device('cuda')

        # Config
        yaml = YAML(typ='rt')
        config = yaml.load(open('mPLUG/configs/retrieval_flickr30k_mplug_large.yaml', 'r'))
        config['text_encoder'] = 'bert-base-uncased'
        config['bert_config'] = 'mPLUG/configs/config_bert.json'

        # Tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Model
        model = MPLUG(config=config, tokenizer=tokenizer).to(device)
        model.eval()

        # Checkpoint
        checkpoint = torch.load('../AliceMind/mPLUG/mplug_large.pth', map_location='cpu')
        state_dict = checkpoint['model']
        num_patches = int(config["image_res"] * config["image_res"] / (14 * 14))
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
        pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0), pos_embed.unsqueeze(0))
        state_dict['visual_encoder.visual.positional_embedding'] = pos_embed
        pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 768).float())
        pos_embed = resize_pos_embed(state_dict['visual_encoder_m.visual.positional_embedding'].unsqueeze(0), pos_embed.unsqueeze(0))
        state_dict['visual_encoder_m.visual.positional_embedding'] = pos_embed

        for key in list(state_dict.keys()):
            if ('fusion' in key or 'bert' in key) and 'decode' not in key:
                encoder_key = key.replace('fusion.', '').replace('bert.', '')
                state_dict[encoder_key] = state_dict[key]
                del state_dict[key]

        _ = model.load_state_dict(state_dict, strict=False)

        # Preprocess images
        normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])

        with torch.no_grad():
            for image_id, image_data in tqdm(self.data[dataset_name].items()):
                raw_image = self.get_image(image_data)
                image = test_transform(raw_image).to(device)
                image = image.unsqueeze(dim=0)
                image_feat = model.visual_encoder.visual(image, skip_last_layer=True)
                image_feat = model.visn_layer_norm(model.visn_fc(image_feat))
                image_embed = model.vision_proj(image_feat[:, 0, :])[0]
                image_embed = F.normalize(image_embed, dim=-1)
                for caption_ind, caption_data in enumerate(image_data['captions']):
                    caption = caption_data['caption']
                    text_input = tokenizer(caption, padding='max_length', truncation=True, max_length=30, return_tensors="pt").to(device)
                    text_output = model.text_encoder(text_input.input_ids, attention_mask=text_input.attention_mask)
                    text_feat = text_output.last_hidden_state
                    text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))[0]
                    embed_score = image_embed.dot(text_embed).item()
                    self.data[dataset_name][image_id]['captions'][caption_ind]['automatic_metrics']['mPLUGScore'] = embed_score

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
        
        corr_type_to_func = {'pearson': stats.pearsonr, 'spearman': stats.spearmanr, 'kendall_b': stats.kendalltau, 'kendall_c': lambda x,y: stats.kendalltau(x, y, variant='c')}
        corr_type_to_res = {}
        for corr_type, corr_func in corr_type_to_func.items():
            metric_to_corr = {}
            for metric in all_metrics:
                cur_human_rating_list = [human_rating_list[i] for i in range(len(human_rating_list)) if i not in metric_to_missing_inds[metric]]
                cur_metric_score_list = [metric_to_score_list[metric][i] for i in range(len(metric_to_score_list[metric])) if i not in metric_to_missing_inds[metric]]
                metric_to_corr[metric] = corr_func(cur_human_rating_list, cur_metric_score_list)

            res = [(metric, float(metric_to_corr[metric].statistic)) for metric in all_metrics]
            res.sort(key=lambda x:x[1], reverse=True)
            corr_type_to_res[corr_type] = res
        return corr_type_to_res
    
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
        for direction in ['forward', 'backward']:
            sfs = SequentialFeatureSelector(reg, direction=direction)
            sfs.fit(X, y)
            res[direction] = [all_metrics[i] for i in range(len(all_metrics)) if sfs.get_support()[i]]

        return res
