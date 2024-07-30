from rating_datasets.image_path_rating_dataset import ImagePathRatingDataset
import json
from config import coco_image_dir_path
import os

class ThumbDataset(ImagePathRatingDataset):
    def get_name(self):
        return 'thumb'
    
    def collect_data(self):
        data = {}
        human_rating_file_path = 'THumB/mscoco/mscoco_THumB-1.0.jsonl'
        with open(human_rating_file_path, 'r') as fp:
            jl = list(fp)
        samples = [json.loads(x) for x in jl]
        ref_file = 'THumB/mscoco/mscoco_references.json'
        with open(ref_file, 'r') as fp:
            jl = list(fp)
        refs = [json.loads(x) for x in jl]
        iid2refs = {int(x['seg_id']): x['refs'] for x in refs}
        for sample in samples:
            image_id = int(sample['image'].split('.')[0].split('_')[-1])
            if image_id not in data:
                dir = sample['image'].split('_')[1]
                data[image_id] = {
                    'references': iid2refs[image_id],
                    'file_path': os.path.join(coco_image_dir_path, dir, sample["image"]),
                    'captions': []
                }
            caption = sample['hyp']
            human_rating = sample['human_score']
            precision = sample['P']
            recall = sample['R']
            data[image_id]['captions'].append({'caption': caption, 'human_ratings': [human_rating], 'precision': [precision], 'recall': [recall], 'automatic_metrics': {}})

        self.data['coco'] = data
