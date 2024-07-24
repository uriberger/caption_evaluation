from image_path_rating_dataset import ImagePathRatingDataset
import scipy as sp
import os
import random
from config import pascal_image_dir_path, pascal_data_dir_path

system_ind2type = {1: 'M', 2: 'M', 3: 'M', 4: 'M', 5: 'M', 6: 'H', 7: 'R'}

class Pascal50Dataset(ImagePathRatingDataset):
    def get_name(self):
        return 'pascal50'
    
    def collect_data(self):
        iid2file_path = {}
        for dir_name in os.listdir(pascal_image_dir_path):
            dir_path = os.path.join(pascal_image_dir_path, dir_name)
            if not os.path.isdir(dir_path):
                continue
            for file_name in os.listdir(dir_path):
                if not file_name.endswith('.jpg'):
                    continue
                iid = file_name.split('.')[0]
                file_path = os.path.join(dir_path, file_name)
                iid2file_path[iid] = file_path

        conc_data = sp.io.loadmat(f'{pascal_data_dir_path}/consensus_pascal.mat')['triplets']
        pair_pascal = sp.io.loadmat(f'{pascal_data_dir_path}/pair_pascal.mat')
        pair_data = pair_pascal['new_input']
        system_data = pair_pascal['new_data']
        data = {}
        for im_ind in range(4000):
            im_data = pair_data[0, im_ind]
            iid = im_data[0][0].split('.')[0]
            if iid not in data:
                file_path = iid2file_path[iid]
                data[iid] = {
                        'references': [],
                        'file_path': file_path,
                        'captions': []
                    }
            cand1 = str(im_data[1][0])
            type1 = system_ind2type[system_data[im_ind, 0]]
            cand2 = str(im_data[2][0])
            type2 = system_ind2type[system_data[im_ind, 1]]
            votes_per_image = 48
            first_won_count = 0
            for pair_ind in range(votes_per_image):
                conc_ind = im_ind*votes_per_image + pair_ind
                cur_conc_data = conc_data[0, conc_ind]
                cur_cand1 = str(cur_conc_data[1][0][0][0])
                assert cand1 == cur_cand1
                cur_cand2 = str(cur_conc_data[2][0][0][0])
                assert cand2 == cur_cand2
                if cur_conc_data[3][0][0] == 1:
                    first_won_count += 1
                else:
                    assert cur_conc_data[3][0][0] == -1
                cur_ref = str(cur_conc_data[0][0][0][0])
                if len(data[iid]['references']) < votes_per_image:
                    data[iid]['references'].append(cur_ref)
                else:
                    assert cur_ref == data[iid]['references'][pair_ind]
            first_won_perc = first_won_count/votes_per_image
            if first_won_perc > 0.5:
                first_won = True
            elif first_won_perc < 0.5:
                first_won = False
            else: # Like CLIPScore, break ties randomly
                first_won = (random.randint(1, 2) == 1)
            cur_ind = len(data[iid]['captions'])
            data[iid]['captions'].append({'caption': cand1, 'type': type1, 'human_ratings': [int(first_won)], 'pair': cur_ind+1, 'automatic_metrics': {}})
            data[iid]['captions'].append({'caption': cand2, 'type': type2, 'human_ratings': [1-int(first_won)], 'pair': cur_ind, 'automatic_metrics': {}})

        # Following previous work, randomly select 5 references
        for iid, im_data in data.items():
            data[iid]['references'] = random.sample(im_data['references'], 5)
            
        self.data['pascal'] = data
