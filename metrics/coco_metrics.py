from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from collections import OrderedDict

def compute_coco_metrics(candidates, references):
    # Collect references and candidates
    ref_dict = {i: references[i] for i in range(len(references))}
    cand_dict = {i: [candidates[i]] for i in range(len(candidates))}

    # Tokenize
    tokenizer = PTBTokenizer()
    tokenized_references = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in ref_dict.items()})
    tokenized_candidates = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in cand_dict.items()})

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

    return metric_name_to_scores
