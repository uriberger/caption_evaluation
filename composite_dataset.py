from human_rating_dataset import HumanRatingDataset
import os
import csv
import json

iid2file_path = {
    'flickr8k': lambda x: f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x}.jpg',
    'flickr30k': lambda x: f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x}.jpg',
    'coco': lambda x: f'/cs/labs/oabend/uriber/datasets/COCO/{x.split("_")[1]}/{x}'
}
entry2iid = {
    'flickr8k': lambda x: x[27].split('/')[-1].split('_')[0],
    'flickr30k': lambda x: x[27].split('/')[-1].split('.')[0],
    'coco': lambda x: x[27].split('/')[-1]
}
dataset2caption_num = {
    'flickr8k': 3,
    'flickr30k': 4,
    'coco': 4
}
dataset_to_file_name = {'flickr8k': '8k', 'flickr30k': '30k', 'coco': 'coco'}
human_rating_dir = '/cs/labs/oabend/uriber/datasets/AMT_eval'

class CompositeDataset(HumanRatingDataset):

    def collect_data(self):
        datasets = ['flickr8k', 'flickr30k', 'coco']

        for dataset in datasets:
            self.collect_data_for_dataset(dataset)

        res_str = 'Collected '
        first = True
        for dataset in datasets:
            if first:
                first = False
            else:
                res_str += ', '
            res_str += f'{len(self.data[dataset])} samples for {dataset}'
        print(res_str, flush=True)

    def collect_data_for_dataset(self, dataset_name):
        human_rating_file_path = os.path.join(human_rating_dir, f'{dataset_to_file_name[dataset_name]}_correctness.csv')

        if dataset_name.startswith('flickr'):
            with open('/cs/labs/oabend/uriber/datasets/flickr30/karpathy/dataset_flickr30k.json', 'r') as fp:
                json_data = json.load(fp)['images']
            data = {x['imgid']: {'references': [y['raw'] for y in x['sentences']]} for x in json_data}
        elif dataset_name == 'coco':
            with open('/cs/snapless/gabis/uriber/CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
                json_data = json.load(fp)['images']
            data = {x['cocoid']: {'references': [y['raw'] for y in x['sentences']]} for x in json_data}

        for image_id in data.keys():
            image_file_path = iid2file_path[dataset_name](image_id)
            if not os.path.isfile(image_file_path):
                continue
            data[image_id]['file_path'] = image_file_path
            data[image_id]['captions'] = []
        
        with open(human_rating_file_path, 'r') as fp:
            my_reader = csv.reader(fp, delimiter=';')
            first = True
            for sample in my_reader:
                if first:
                    first = False
                    continue
                image_id_str = entry2iid[dataset_name](sample)
                if len(image_id_str) == 0:
                    continue
                image_id = int(image_id_str)
                if image_id not in data:
                    continue
                cap_num = dataset2caption_num[dataset_name]
                for i in range(cap_num):
                    data[image_id]['captions'].append({'caption': sample[28+i], 'rating': sample[28+cap_num+i]})

        self.data[dataset_name] = data
