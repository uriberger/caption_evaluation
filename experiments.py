from copy import deepcopy
import random
from tqdm import tqdm
from composite_dataset import CompositeDataset
from flickr8k_expert_dataset import Flickr8kDataset
from thumb_dataset import ThumbDataset
from polaris_dataset import PolarisDataset

def test_sampling_effect(hr_dataset, fraction):
    corr_type_to_res = hr_dataset.compute_correlation(plot=False)

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

    corr_type_to_max_diff = {}
    for corr_type, res in corr_type_to_res.items():
        temp_res = temp_corr_type_to_res[corr_type]
        # if [x[0] for x in res] != [x[0] for x in temp_res]:
        #     print(f'Ranking for {corr_type} changed')
        metric_to_corr = {x[0]: x[1] for x in res}
        temp_metric_to_corr = {x[0]: x[1] for x in temp_res}
        # max_diff = max([abs(1 - x[1]/metric_to_corr[x[0]]) for x in temp_metric_to_corr.items()])
        max_diff = max([abs(x[1] - metric_to_corr[x[0]]) for x in temp_metric_to_corr.items()])
        corr_type_to_max_diff[corr_type] = max_diff
    
    return corr_type_to_max_diff

def sampling_effect_multiple_trials(hr_dataset, fraction):
    trial_num = 10
    passed_all = True
    for _ in tqdm(range(trial_num), desc=f'Experiments on {hr_dataset.get_name()}'):
        corr_type_to_max_diff = test_sampling_effect(hr_dataset, fraction)
        if max(corr_type_to_max_diff.values()) >= 0.01:
            passed_all = False
            break
    return passed_all

def test_all_sampling_effects(fraction):
    dataset = CompositeDataset()
    dataset.load()
    print(f'Composite dataset: {sampling_effect_multiple_trials(dataset, fraction)}')

    dataset = Flickr8kDataset('expert')
    dataset.load()
    print(f'Flickr8k expert: {sampling_effect_multiple_trials(dataset, fraction)}')

    dataset = Flickr8kDataset('crowd_flower')
    dataset.load()
    print(f'Flickr8k crowd flower: {sampling_effect_multiple_trials(dataset, fraction)}')

    dataset = ThumbDataset()
    dataset.load()
    print(f'Thumb: {sampling_effect_multiple_trials(dataset, fraction)}')

    dataset = PolarisDataset()
    dataset.load()
    print(f'Polaris expert: {sampling_effect_multiple_trials(dataset, fraction)}')
