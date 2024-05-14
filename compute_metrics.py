from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
from evaluate import load
import statistics
import numpy as np
from collections import defaultdict

def remap_image_ids(references, candidates):
    # Image ids can be too long for SPICE. Map them to small integer ids
    new_to_orig_id = list(set([x['image_id'] for x in references]))
    orig_to_new_id = {new_to_orig_id[i]: i for i in range(len(new_to_orig_id))}
    modified_refs = defaultdict(list)
    for x in references:
        modified_refs[orig_to_new_id[x['image_id']]].append(x['caption'])
    candidates = {orig_to_new_id[x['image_id']]: [x['caption']] for x in candidates}

    return modified_refs, candidates

def compute_metrics(references, candidates):
    references, candidates = remap_image_ids(references, candidates)
    tokenizer = PTBTokenizer()
    tokenized_references = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in references.items()})
    tokenized_candidates = tokenizer.tokenize({x[0]: [{'caption': y} for y in x[1]] for x in candidates.items()})

    ###BLEU#####
    print("Compute BLEU ... ")
    pycoco_bleu = Bleu()
    bleu, _ = pycoco_bleu.compute_score(tokenized_references, tokenized_candidates)

    ####METEOR###
    print("Compute METEOR ... ")
    pycoco_meteor = Meteor()
    meteor, _ = pycoco_meteor.compute_score(tokenized_references, tokenized_candidates)
    del pycoco_meteor

    ####ROUGE###
    print("Compute ROUGE ... ")
    pycoco_rouge = Rouge()
    rouge, _ = pycoco_rouge.compute_score(tokenized_references, tokenized_candidates)

    ####CIDER###
    print("Compute CIDER ... ")
    pycoco_cider = Cider()
    cider, _ = pycoco_cider.compute_score(tokenized_references, tokenized_candidates)

    ####SPICE###
    print("Compute SPICE ... ")
    pycoco_spice = Spice()
    spice, spice_scores = pycoco_spice.compute_score(tokenized_references, tokenized_candidates)
    spice_submetrics = ['Relation', 'Cardinality', 'Attribute', 'Size', 'Color', 'Object']
    spice_submetrics_res = {}
    for submetric in spice_submetrics:
        spice_submetrics_res[submetric] = np.mean(np.array([x[submetric]['f'] for x in spice_scores if not np.isnan(x[submetric]['f'])]))

    ####BERTScore###
    bertscore = load("bertscore")
    reference_list = list(references.items())
    reference_list.sort(key=lambda x:x[0])
    references = [x[1] for x in reference_list]
    prediction_list = list(candidates.items())
    prediction_list.sort(key=lambda x:x[0])
    predictions = [x[1][0] for x in prediction_list]
    results = bertscore.compute(predictions=predictions, references=references, lang=lang)
    bertscore = statistics.mean(results['f1'])

    res = {'bleu1': bleu[0], 'bleu2': bleu[1], 'bleu3': bleu[2], 'bleu4': bleu[3], 'rouge': rouge, 'cider': cider, 'bertscore': bertscore}
    if lang == 'en' and spice:
        res['spice'] = spice
        for submetric, val in spice_submetrics_res.items():
            res[submetric] = val
    if meteor:
        res['meteor'] = meteor
    return res
