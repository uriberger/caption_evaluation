from copy import deepcopy
import random

def test_sampling_effect(hr_dataset, fraction):
    corr_type_to_res = hr_dataset.compute_correlation()

    # Build temp dataset
    temp_hr_dataset = deepcopy(hr_dataset)
    all_samples = [(x[0], y[0], z) for x in hr_dataset.data.items() for y in x[1].items() for z in range(len(y[1]['captions']))]
    sample_size = int(fraction*len(all_samples))
    chosen_samples = random.sample(all_samples, sample_size)
    temp_hr_dataset.data = {}
    for dataset_name, image_id, caption_ind in chosen_samples:
        if dataset_name not in temp_hr_dataset.data:
            temp_hr_dataset.data[dataset_name] = {}
        if image_id not in temp_hr_dataset.data[dataset_name]:
            temp_hr_dataset.data[dataset_name][image_id] = {'captions': []}
        temp_hr_dataset.data[dataset_name][image_id]['captions'].append(hr_dataset.data[dataset_name][image_id]['captions'][caption_ind])

    temp_corr_type_to_res = temp_hr_dataset.compute_correlation()

    for corr_type, res in corr_type_to_res.items():
        # First check if the ranking has changed
        temp_res = temp_corr_type_to_res[corr_type]
        if [x[0] for x in res] != [x[0] for x in temp_res]:
            print(f'Ranking for {corr_type} changed')
        metric_to_corr = {x[0]: x[1] for x in res}
        temp_metric_to_corr = {x[0]: x[1] for x in temp_res}
        # max_diff = max([abs(1 - x[1]/metric_to_corr[x[0]]) for x in temp_metric_to_corr.items()])
        max_diff = max([abs(x[1] - metric_to_corr[x[0]]) for x in temp_metric_to_corr.items()])
        print(f'Max diff in {corr_type}: {max_diff}')
