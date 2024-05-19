from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from evaluate import load
from collections import OrderedDict

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
