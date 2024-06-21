from human_rating_dataset import HumanRatingDataset
from collections import defaultdict
import os

flickr8k_dir = 'flickr8k_data'
class Flickr8kDataset(HumanRatingDataset):
    def __init__(self, name):
        super(Flickr8kDataset, self).__init__()
        self.name = name

    def get_name(self):
        return f'flickr8k_{self.name}'
        
    def get_file_name2iid_func(self, dataset_name):
        return lambda x: int(x)

    def collect_data(self):
        iid2captions = defaultdict(list)
        with open(f'{flickr8k_dir}/Flickr8k.token.txt', 'r') as fp:
            for line in fp:
                line_parts = line.strip().split('\t')
                assert len(line_parts) == 2
                image_id = int(line_parts[0].split('_')[0])
                caption = line_parts[1]
                iid2captions[image_id].append(caption)

        data = {}
        human_rating_file_name = 'ExpertAnnotations' if self.name == 'expert' else 'CrowdFlowerAnnotations'
        with open(f'{flickr8k_dir}/{human_rating_file_name}.txt', 'r') as fp:
            for line in fp:
                line_parts = line.strip().split('\t')
                image_id = int(line_parts[0].split('_')[0])
                if image_id not in data:
                    file_path = f'/cs/labs/oabend/uriber/datasets/flickr30/images/{image_id}.jpg'
                    if not os.path.isfile(file_path):
                        continue
                    data[image_id] = {
                        'references': iid2captions[image_id],
                        'file_path': file_path,
                        'captions': []
                        }
                caption_image_id = int(line_parts[1].split('_')[0])
                caption_ind = int(line_parts[1].split('#')[1])
                caption = iid2captions[caption_image_id][caption_ind]
                human_ratings = [float(x) for x in line_parts[2:]]
                human_rating = sum(human_ratings)/len(human_ratings)
                data[image_id]['captions'].append({'caption': caption, 'human_rating': human_rating, 'automatic_metrics': {}})
        
        self.data['flickr8k'] = data
