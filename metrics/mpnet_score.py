import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import statistics

def numpy_cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def compute_mpnet_score(candidates, references, agg_method='mean'):
    model = SentenceTransformer('all-mpnet-base-v2')
    model.eval()

    scores = []
    for candidate, refs in zip(candidates, references):
        with torch.no_grad():
            cand_embedding = model.encode(candidate)
            ref_embeddings = [model.encode(ref) for ref in refs]
        cur_scores = [numpy_cosine_similarity(cand_embedding, ref_embedding).item() for ref_embedding in ref_embeddings]
        if agg_method == 'mean':
            score = statistics.mean(cur_scores)
        elif agg_method == 'max':
            score = max(cur_scores)
        scores.append(score)

    return scores