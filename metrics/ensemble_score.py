import json

reference_metrics = ['Exact noun overlap', 'Fuzzy noun overlap', 'Exact verb overlap', 'Fuzzy verb overlap', 'RefCLIPScore', 'METEOR', 'ROUGE', 'RefPACScore', 'SPICE', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'CIDEr', 'MPNetScore', 'polos']
image_metrics = ['CLIPScore', 'RefCLIPScore', 'PACScore', 'RefPACScore', 'BLIP2Score', 'CLIPImageScore', 'polos']

def compute_ensemble_score(candidates, references, image_paths, weights=None):
    if references is not None:
        assert len(candidates) == len(references), 'Please provide the same number of candidates and reference sets'
    if image_paths is not None:
        assert len(candidates) == len(image_paths), 'Please provide the same number of candidates and image paths'

    if weights is None:
        with open('ensemble_weights.json', 'r') as fp:
            weights = json.load(fp)

    reference_metrics_in_weights = set(reference_metrics).intersection(weights.keys())
    if references is None and len(reference_metrics_in_weights) > 0:
        assert False, f'The following metrics requires references, but you provide none: {", ".join(list(reference_metrics_in_weights))}'

    image_metrics_in_weights = set(image_metrics).intersection(weights.keys())
    if image_paths is None and len(image_metrics_in_weights) > 0:
        assert False, f'The following metrics requires iamge paths, but you provide none: {", ".join(list(image_metrics_in_weights))}'

    metric2scores = {}
    for metric_name in weights.keys():
        if metric_name not in metric2scores:
            if metric_name in ['METEOR', 'ROUGE', 'SPICE', 'BLEU1', 'BLEU2', 'BLEU3', 'BLEU4', 'CIDEr']:
                from metrics.coco_metrics import compute_coco_metrics
                coco_scores = compute_coco_metrics(candidates, references)
                for coco_metric_name, coco_metric_scores in coco_scores.items():
                    metric2scores[coco_metric_name] = coco_metric_scores
            elif metric_name in ['CLIPScore', 'RefCLIPScore']:
                from metrics.clipscore import compute_clipscore
                clipscores, refclipscores = compute_clipscore(candidates, references, image_paths)
                metric2scores['CLIPScore'] = clipscores
                metric2scores['RefCLIPScore'] = refclipscores
            elif metric_name in ['PACScore', 'RefPACScore']:
                from metrics.pacscore import compute_pacscore
                pacscores, refpacscores = compute_pacscore(candidates, references, image_paths)
                metric2scores['PACScore'] = pacscores
                metric2scores['RefPACScore'] = refpacscores
            elif metric_name == 'BLIP2Score':
                from metrics.blip2score import compute_blip2score
                blip2scores = compute_blip2score(candidates, image_paths)
                metric2scores[metric_name] = blip2scores
            elif metric_name == 'CLIPImageScore':
                from metrics.clip_image_score import compute_clip_image_score
                clip_image_scores = compute_clip_image_score(candidates, image_paths)
                metric2scores[metric_name] = clip_image_scores
            elif metric_name == 'polos':
                from metrics.polos import compute_polos
                polos_scores = compute_polos(candidates, references, image_paths)
                metric2scores[metric_name] = polos_scores
            elif metric_name == 'MPNetScore':
                from metrics.mpnet_score import compute_mpnet_score
                mpnet_scores = compute_mpnet_score(candidates, references)
                metric2scores[metric_name] = mpnet_scores

    ensemble_scores = [sum([x[1]*metric2scores[x[0]][i] for x in weights.items()]) for i in range(len(candidates))]
    return ensemble_scores
