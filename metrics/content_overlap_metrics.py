from content_score import compute_and_add_content_recall

def compute_content_overlap_metrics(candidates, references):
    samples = [{'candidate_summary': candidate, 'refs': refs} for candidate, refs in zip(candidates, references)]

    res = compute_and_add_content_recall(samples, 'refs')
    return {
        'Exact noun overlap': [x['scores']['content_recall']['candidate_summary_noun_recall'] for x in res],
        'Fuzzy noun overlap': [x['scores']['content_recall']['candidate_summary_noun_fuzzy_recall'] for x in res],
        'Exact verb overlap': [x['scores']['content_recall']['candidate_summary_verb_recall'] for x in res],
        'Fuzzy verb overlap': [x['scores']['content_recall']['candidate_summary_verb_fuzzy_recall'] for x in res]
    }
