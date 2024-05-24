from nubia_score import Nubia
from composite_dataset import CompositeDataset
from tqdm import tqdm
import pickle

self = CompositeDataset()
self.load()

res = {}

dataset_name = 'coco'
nubia = Nubia()
for image_id, image_data in tqdm(list(self.data[dataset_name].items())):
    res[image_id] = []
    for caption_ind, caption_data in enumerate(image_data['captions']):
        ignore_refs = []
        if 'ignore_refs' in caption_data:
            ignore_refs = caption_data['ignore_refs']
        references = [image_data['references'][i] for i in range(len(image_data['references'])) if i not in ignore_refs]
        candidate = caption_data['caption']
        scores = [nubia.score(x, candidate) for x in references]
        res[image_id].append(scores)

with open(f'{dataset_name}_nubia_scores.pkl', 'wb') as fp:
    pickle.dump(res, fp)

