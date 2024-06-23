from human_rating_dataset import HumanRatingDataset
from datasets import load_dataset
from tqdm import tqdm
import os
import shutil

image_dir = 'polaris_temp'

class PolarisDataset(HumanRatingDataset):
    def __init__(self):
        super(PolarisDataset, self).__init__()
        self.hash2image_obj = None

    def get_name(self):
        return 'polaris'
    
    def get_file_name2iid_func(self, dataset_name):
        return lambda x: int(x)

    def collect_data(self):
        dataset = load_dataset('yuwd/Polaris')
        data = {}
        for key, value in dataset.items():
            data[key] = {}
            for sample in tqdm(value, desc=f'Collecting {key} data'):
                image_id = self.hash_func(sample['refs'])
                if image_id not in data[key]:
                    data[key][image_id] = {
                        'references': sample['refs'],
                        'captions': []
                    }
                data[key][image_id]['captions'].append({'caption': sample['cand'], 'human_rating': sample['human_score'], 'automatic_metrics': {}})

        self.data = data
    
    def init_hash2image_obj(self):
        dataset = load_dataset('yuwd/Polaris')
        self.hash2image_obj = {}
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)
        for key, value in dataset.items():
            self.hash2image_obj[key] = {}
            for sample in tqdm(value, desc=f'Building {key} hash2image_obj'):
                cur_hash = self.hash_func(sample['refs'])
                if cur_hash in self.hash2image_obj[key]:
                    assert self.hash2image_obj[key][cur_hash] == sample['img']
                else:
                    self.hash2image_obj[key][cur_hash] = sample['img']

    def hash_func(self, ref_list):
        return sum([ord(x[0]) + 2*ord(x[1]) - 3*ord(x[2])**2 + 4*ord(x[3])**3 + ord(x[-1])**3 - ord(x[-2])**2 + 6*ord(x[-3])**3 for x in ref_list])

    def get_image(self, dataset_name, image_data):
        if self.hash2image_obj is None:
            self.init_hash2image_obj()

        cur_hash = self.hash_func(image_data['references'])
        return self.hash2image_obj[dataset_name][cur_hash]
    
    def get_file_path(self, dataset_name, image_data):
        cur_hash = self.hash_func(image_data['references'])
        file_name = str(cur_hash)
        file_path = os.path.join(image_dir, file_name + '.jpg')
        if not os.path.isfile(file_path):
            image_obj = self.get_image(dataset_name, image_data)
            image_obj.save(file_path)
        return file_path
    
    def clean_temp_files(self):
        shutil.rmtree(image_dir)
