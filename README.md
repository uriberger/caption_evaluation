# Image Captioning Evaluation: A Comprehensive Survey and Novel Ensemble Method

## Installation
Python version: 3.9.2

1. Run:
```
bash install.sh
```

2. Download the pacscore clip_ViT-B-32.pth fine-tuned checkpoint from the [pacscore repo](https://github.com/aimagelab/pacscore) and locate it under pacscore/checkpoints.

## Ensemble captioning evaluation
To use the ensemble evaluation method with our proposed weights, use:
```
from metrics.ensemble_score import compute_ensemble_score

scores = compute_ensemble_score(candidates, references, image_paths)
```
Where `candidates` is a list of captions, `references` is a list of lists of reference captions, `image_paths` is a list of strings with locations of images. For example:
```
import json
from metrics.ensemble_score import compute_ensemble_score

with open('example/references.json', 'r') as fp:
    ref_dict = json.load(fp)

with open('example/candidates.json', 'r') as fp:
    cand_dict = json.load(fp)

correct_candidates = [cand_dict['im1'], cand_dict['im2']]
references = [ref_dict['im1'], ref_dict['im2']]
image_paths = ['example/im1.jpg', 'example/im2.jpg']

scores = compute_ensemble_score(correct_candidates, references, image_paths) # 0.9107, 0.9712

incorrect_candidates = [cand_dict['im2'], cand_dict['im1']]
scores = compute_ensemble_score(incorrect_candidates, references, image_paths) # 0.2022, 0.2118
```
To use your own weights, provide a mapping from metric to weight using the `weights` arguments:
```
scores = compute_ensemble_score(candidates, references, image_paths, weights=your_weights)
```
If none of the metrics in your provided weights use references or images, you can specify `None` in the corresponding argument.

## Correlation with human ratings
To replicate the results for correlation with human ratings provided in the paper, first download the relevant data, and then use our code to run the experiments.

### Data
#### Image captioning datasets
Each human ratings datasets requires data from existing image captioning datasets. First, download the following image captioning datasets:
- Flickr8k (needed for Flickr8k-Expert, Flickr8k-CF and Composite): Download the dataset from the [following link](https://www.kaggle.com/datasets/sayanf/flickr8k).
- Flickr30k (needed for Composite and the Reformulations dataset): Download the dataset from the [following link](https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset).
- MSCOCO (needed for Composite, ThuMB and the Reformulations dataset): Download the 2014 val images from the [following link](https://cocodataset.org/#download).
- For both MSCOCO and Flickr30k, download the Karpathy split json files (dataset_coco.json and dataset_flickr30k.json) from the [following link](https://www.kaggle.com/datasets/shtvkumar/karpathy-splits).
- Pascal (needed for the Pascal50S dataset): Download the 2008 training/validation images from the [following link](http://host.robots.ox.ac.uk/pascal/VOC/voc2008/index.html#devkit).
#### Human rating datasets
- Composite: Download the AMT results from the [following link](https://imagesdg.wordpress.com/image-to-scene-description-graph/).
- ThuMB: Clone the ThuMB repo under the project root: `git clone https://github.com/jungokasai/THumB.git`
- Pascal50S: Downloaded the consensus ratings from the [following link](https://vrama91.github.io/cider/).
- The reformulations dataset is currently included in this github repository, for anonymity.

After downloading all of the above, unzip and untar where necessary, and update the relevant paths in the config.py file.

### Run experiments
To compute the correlation with human ratings, run the following:
```
python run_experiments.py --dataset [flickr8k_expert/flickr_cf/composite/thumb/polaris]
```
To compute the pairwise accuracy, run the following:
```
python run_experiments.py --dataset [pascal50/reformulations] --eval_method pairwise
```
Note: the CLIPImageScore metric is highly time consuming, and is by default not included in this scripts. If you do wish to use it (acknowledging that it would take very long), use the `--clip_image_score` flag:
```
python run_experiments.py --dataset [flickr8k_expert/flickr_cf/composite/thumb/polaris] --clip_image_score
python run_experiments.py --dataset [pascal50/reformulations] --eval_method pairwise --clip_image_score
```

## Metric usage analysis
To replicate the analysis and plots regarding metric usage analysis from the paper, run the following:
```
python metric_usage_analysis.py
```
This will generate figures 2,3,5 from the paper and will dump them in the project directory.
