from image_path_rating_dataset import ImagePathRatingDataset
import json

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
        with open('../AliceMind/mPLUG/reformulation_data/full_reformulation_dataset.json', 'r') as fp:
            samples = json.load(fp)

        with open('../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_orig_data = json.load(fp)['images']
        with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
            flickr_orig_data = json.load(fp)['images']
        dataset2iid2orig_data = {
            'coco': {x['cocoid']: x for x in coco_orig_data},
            'flickr30k': {file_name2iid['flickr30k'](x['filename'].split('.')[0]): x for x in flickr_orig_data}
            }

        data = {'coco': {}, 'flickr30k': {}}
        for sample in samples:
            cur_dataset = 'coco' if 'COCO' in sample['image'] else 'flickr30k'
            image_id = file_name2iid[cur_dataset](sample['image'].split('.')[0])
            if image_id not in data[cur_dataset]:
                cur_orig_data = dataset2iid2orig_data[cur_dataset][image_id]
                dataset_dir = 'COCO' if cur_dataset == 'coco' else 'flickr30'
                data[cur_dataset][image_id] = {
                    'references': [x['raw'] for x in cur_orig_data['sentences']],
                    'file_path': f'/cs/labs/oabend/uriber/datasets/{dataset_dir}/{cur_orig_data["filepath"]}/{cur_orig_data["filename"]}',
                    'captions': []
                }
            cur_ind = len(data[cur_dataset][image_id]['captions'])
            data[cur_dataset][image_id]['captions'].append({'caption': sample['question'], 'human_rating': 0, 'tag': 'before', 'pair': cur_ind+1, 'automatic_metrics': {}})
            data[cur_dataset][image_id]['captions'].append({'caption': sample['answer'], 'human_rating': 1, 'tag': 'after', 'pair': cur_ind, 'automatic_metrics': {}})
