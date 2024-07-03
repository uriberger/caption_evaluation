from image_path_rating_dataset import ImagePathRatingDataset
import json
from Levenshtein import distance
import numpy as np

class ThumbDataset(ImagePathRatingDataset):
    def get_name(self):
        return 'thumb'
        
    def get_file_name2iid_func(self, dataset_name):
        return lambda x: int(x.split('_')[-1])
    
    def collect_data(self):
        data = {}
        human_rating_file_path = 'THumB/mscoco/mscoco_THumB-1.0.jsonl'
        with open(human_rating_file_path, 'r') as fp:
            jl = list(fp)
        samples = [json.loads(x) for x in jl]
        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_orig_data = json.load(fp)['images']
        iid2orig_data = {x['cocoid']: x for x in coco_orig_data if x['split'] == 'test'}
        for sample in samples:
            image_id = int(sample['image'].split('.')[0].split('_')[-1])
            if image_id not in data:
                cur_orig_data = iid2orig_data[image_id]
                data[image_id] = {
                    'references': [x['raw'] for x in cur_orig_data['sentences']],
                    'file_path': f'/cs/labs/oabend/uriber/datasets/COCO/{cur_orig_data["filepath"]}/{sample["image"]}',
                    'captions': []
                }
            if len(data[image_id]['captions']) == 4:
                # Fifth caption in this dataset is one of the references. Following CLIPScore, ignore this candidate
                continue
            caption = sample['hyp']
            human_rating = sample['human_score']
            data[image_id]['captions'].append({'caption': caption, 'human_rating': human_rating, 'automatic_metrics': {}})
            if len(data[image_id]['captions']) == 5:
                # If we didn't ignore, tell the ref based metrics to ignore the same reference
                data[image_id]['captions'][-1]['ignore_refs'] = [np.argmin([distance(caption, ref) for ref in data[image_id]['references']])]

        self.data['coco'] = data
