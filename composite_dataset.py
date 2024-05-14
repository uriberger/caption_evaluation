from human_rating_dataset import HumanRatingDataset
import os
import csv

class CompositeDataset(HumanRatingDataset):

    def collect_data(self):
        self.collect_data_for_dataset('flickr8k')
        self.collect_data_for_dataset('flickr30k')
        self.collect_data_for_dataset('coco')

    def collect_data_for_dataset(self, dataset_name):
        human_rating_dir = '/cs/labs/oabend/uriber/datasets/AMT_eval'
        dataset_to_file_name = {'flickr8k': '8k', 'flickr30k': '30k', 'coco': 'coco'}

        human_rating_file_path = os.path.join(human_rating_dir, f'{dataset_to_file_name[dataset_name]}_correctness.csv')
        iid2file_path = {
            'flickr8k': lambda x: f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x}.jpg',
            'flickr30k': lambda x: f'/cs/labs/oabend/uriber/datasets/flickr30/images/{x}.jpg',
            'coco': lambda x: f'/cs/labs/oabend/uriber/datasets/COCO/{x.split("_")[1]}/{x}'
        }
        entry2iid = {
            'flickr8k': lambda x: x[27].split('/')[-1].split('_')[0],
            'flickr30k': lambda x: x[27].split('/')[-1].split('_')[0],
            'coco': lambda x: x[27].split('/')[-1]
        }
        dataset2caption_num = {
            'flickr8k': 3,
            'flickr30k': 4,
            'coco': 4
        }
        
        with open(human_rating_file_path, 'r') as fp:
            my_reader = csv.reader(fp, delimiter=';')
            first = True
            for sample in my_reader:
                if first:
                    first = False
                    continue
                image_id = entry2iid[dataset_name](sample)
                image_file_path = iid2file_path[dataset_name](image_id)
                cap_num = dataset2caption_num[dataset_name]
                for i in range(cap_num):
                    self.data.append({'image_path': image_file_path, 'caption': sample[28+i], 'rating': sample[28+cap_num+i]})
