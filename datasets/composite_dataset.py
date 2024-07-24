from datasets.image_path_rating_dataset import ImagePathRatingDataset
import os
import csv
import json
from Levenshtein import distance
from collections import defaultdict
import numpy as np
from config import flickr30k_image_dir_path, \
    flickr8k_path, \
    coco_image_dir_path, \
    composite_path, \
    flickr30k_json_path, \
    coco_json_path

entry2iid = {
    'flickr8k': lambda x: x[27].split('/')[-1].split('_')[0],
    'flickr30k': lambda x: x[27].split('/')[-1].split('.')[0],
    'coco': lambda x: x[27].split('/')[-1].split('_')[-1].split('.')[0]
}
dataset2caption_num = {
    'flickr8k': 3,
    'flickr30k': 4,
    'coco': 4
}
dataset_to_file_name = {'flickr8k': '8k', 'flickr30k': '30k', 'coco': 'coco'}
human_rating_dir = composite_path

class CompositeDataset(ImagePathRatingDataset):
    def get_name(self):
        return 'composite'
    
    def collect_data(self, remove_gt_candidates=False, remove_gt_candidates_from_ref_set=True):
        splits = ['flickr8k', 'flickr30k', 'coco']

        for split in splits:
            self.collect_data_for_split(split, remove_gt_candidates, remove_gt_candidates_from_ref_set)

        res_str = 'Collected '
        first = True
        for split in splits:
            if first:
                first = False
            else:
                res_str += ', '
            res_str += f'{len(self.data[split])} images and {len([x for outer in self.data[split].values() for x in outer["captions"]])} captions for {split}'
        print(res_str, flush=True)

    def collect_data_for_split(self, split_name, remove_gt_candidates, remove_gt_candidates_from_ref_set):
        human_rating_file_path = os.path.join(human_rating_dir, f'{dataset_to_file_name[split_name]}_correctness.csv')

        if split_name == 'flickr8k':
            iid2refs = defaultdict(list)
            with open(os.path.join(flickr8k_path, 'Flickr8k_text', 'Flickr8k.token.txt'), 'r') as fp:
                for line in fp:
                    iid = line.split('.jpg#')[0]
                    ref = line.strip().split('\t')[1]
                    iid2refs[iid].append(ref)
            data = {int(x[0].split('_')[0]): {
                'references': x[1],
                'file_path': os.path.join(flickr8k_path, 'Flickr8k_Dataset', f'{x[0]}.jpg'),
                'captions': []
                } for x in iid2refs.items()}
        elif split_name == 'flickr30k':
            with open(flickr30k_json_path, 'r') as fp:
                json_data = json.load(fp)['images']
            data = {int(x['filename'].split('.')[0]): {
                'references': [y['raw'] for y in x['sentences']],
                'file_path': os.path.join(flickr30k_image_dir_path, f'{x["filename"]}'),
                'captions': []
                } for x in json_data}
        elif split_name == 'coco':
            with open(coco_json_path, 'r') as fp:
                json_data = json.load(fp)['images']
            data = {x['cocoid']: {
                'references': [y['raw'] for y in x['sentences']],
                'file_path': os.path.join(coco_image_dir_path, f'{x["filepath"]}/{x["filename"]}'),
                'captions': []
                } for x in json_data}

        with open(human_rating_file_path, 'r') as fp:
            my_reader = csv.reader(fp, delimiter=';')
            first = True
            for sample in my_reader:
                if first:
                    first = False
                    continue
                image_id_str = entry2iid[split_name](sample)
                if len(image_id_str) == 0:
                    continue
                image_id = int(image_id_str)
                if image_id not in data:
                    print(f'Missing image in {split_name}')
                    continue
                cap_num = dataset2caption_num[split_name]
                for i in range(cap_num):
                    if i == 3:
                        continue # All previous studies only used the first 3 captions
                    if remove_gt_candidates and i == 0:
                        # First caption in this dataset is one of the references, like in the CLIPScore paper: ignore
                        continue
                    data[image_id]['captions'].append({'caption': sample[28+i], 'human_ratings': [int(sample[28+cap_num+i])], 'automatic_metrics': {}})

        data = {x[0]: x[1] for x in data.items() if len(x[1]['captions']) > 0}

        if remove_gt_candidates_from_ref_set:
            # Remove candidates from the reference set
            for image_id, image_data in data.items():
                cand_ind_in_refs = np.argmin([distance(image_data['captions'][0]['caption'], ref) for ref in data[image_id]['references']])
                ref_set = data[image_id]['references']
                data[image_id]['references'] = [ref_set[i] for i in range(len(ref_set)) if i != cand_ind_in_refs]

        self.data[split_name] = data
