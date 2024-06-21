from human_rating_dataset import HumanRatingDataset
from PIL import Image

class ImagePathRatingDataset(HumanRatingDataset):
    def get_image(image_data):
        return Image.open(image_data['file_path']).convert("RGB")
    
    def get_file_path(self, dataset_name, image_data):
        return image_data['file_path']
