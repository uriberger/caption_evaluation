from rating_datasets.image_path_rating_dataset import ImagePathRatingDataset
from collections import defaultdict
import os
from config import flickr8k_path

class Flickr8kDataset(ImagePathRatingDataset):
    def __init__(self, name):
        super(Flickr8kDataset, self).__init__()
        self.name = name

    def get_name(self):
        return f'flickr8k_{self.name}'

    def collect_data(self):
        iid2captions = defaultdict(list)
        with open(os.path.join(flickr8k_path, 'Flickr8k_text', 'Flickr8k.token.txt'), 'r') as fp:
            for line in fp:
                line_parts = line.strip().split('\t')
                assert len(line_parts) == 2
                image_id = int(line_parts[0].split('_')[0])
                caption = line_parts[1]
                iid2captions[image_id].append(caption)

        data = {}
        human_rating_file_name = 'ExpertAnnotations' if self.name == 'expert' else 'CrowdFlowerAnnotations'
        with open(os.path.join(flickr8k_path, 'Flickr8k_text', f'{human_rating_file_name}.txt'), 'r') as fp:
            for line in fp:
                line_parts = line.strip().split('\t')
                image_id = line_parts[0].split('.')[0]
                if image_id not in data:
                    file_path = os.path.join(flickr8k_path, 'Flickr8k_Dataset', f'{image_id}.jpg')
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
                if self.name == 'expert' and caption in data[image_id]['references']:
                    continue # Similar to the SPICE paper and CLIPSCore paper
                if self.name == 'expert':
                    human_ratings = [float(x) for x in line_parts[2:]]
                else:
                    human_ratings = [float(line_parts[2])]
                data[image_id]['captions'].append({'caption': caption, 'human_ratings': human_ratings, 'automatic_metrics': {}})

        # Remove candidates from the reference set
        for image_id, image_data in data.items():
            candidates = [x['caption'] for x in image_data['captions']]
            data[image_id]['references'] = [x for x in data[image_id]['references'] if x not in candidates]
        
        self.data['flickr8k'] = data
