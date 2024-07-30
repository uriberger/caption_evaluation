from rating_datasets.image_path_rating_dataset import ImagePathRatingDataset
import json
import os
from config import reformulations_json_path, coco_json_path, flickr30k_json_path, coco_image_dir_path, flickr30k_image_dir_path

file_name2iid = {
    'flickr30k': lambda x: int(x),
    'coco': lambda x: int(x.split('_')[-1])
}

class ReformulationsDataset(ImagePathRatingDataset):
    def get_name(self):
        return 'reformulations'
        
    def get_file_name2iid_func(self, dataset_name):
        return file_name2iid[dataset_name]
    
    def collect_data(self):
        with open(reformulations_json_path, 'r') as fp:
            samples = json.load(fp)

        with open(coco_json_path, 'r') as fp:
            coco_orig_data = json.load(fp)['images']
        with open(flickr30k_json_path, 'r') as fp:
            flickr_orig_data = json.load(fp)['images']
        dataset2iid2orig_data = {
            'coco': {x['cocoid']: x for x in coco_orig_data},
            'flickr30k': {file_name2iid['flickr30k'](x['filename'].split('.')[0]): x for x in flickr_orig_data}
            }

        data = {'coco': {}, 'flickr30k': {}}
        for sample in samples:
            if sample['question'].lower() == sample['answer'].lower():
                continue # Remove samples where the original caption is identical to the reformulation
            cur_dataset = 'coco' if 'COCO' in sample['image'] else 'flickr30k'
            image_id = file_name2iid[cur_dataset](sample['image'].split('.')[0].split('/')[-1])
            if image_id not in data[cur_dataset]:
                cur_orig_data = dataset2iid2orig_data[cur_dataset][image_id]
                if cur_dataset == 'coco':
                    split_dir = cur_orig_data["filepath"] if cur_dataset == 'coco' else ''
                    file_dir = os.path.join(coco_image_dir_path, split_dir)
                else:
                    file_dir = flickr30k_image_dir_path
                data[cur_dataset][image_id] = {
                    'references': [x['raw'] for x in cur_orig_data['sentences']],
                    'file_path': os.path.join(file_dir, cur_orig_data['filename']),
                    'captions': []
                }
            cur_ind = len(data[cur_dataset][image_id]['captions'])
            data[cur_dataset][image_id]['captions'].append({'caption': sample['question'], 'human_ratings': [0], 'tag': 'before', 'pair': cur_ind+1, 'automatic_metrics': {}})
            data[cur_dataset][image_id]['captions'].append({'caption': sample['answer'], 'human_ratings': [1], 'tag': 'after', 'pair': cur_ind, 'automatic_metrics': {}})

        self.data = data
