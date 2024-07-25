from lavis.models import load_model_and_preprocess
import imageio.v2 as imageio
import torch
from PIL import Image

def compute_blip2score(candidates, image_paths):
    device = torch.device('cuda')
    model, vis_processors, text_processors = load_model_and_preprocess("blip2_image_text_matching", "pretrain", device=device, is_eval=True)

    prev_image_path = None
    scores = []
    with torch.no_grad():
        for candidate, image_path in zip(candidates, image_paths):
            if image_path != prev_image_path:
                im = imageio.imread(image_path)
                if len(im.shape) != 3 or im.shape[2] != 3:
                    continue
                raw_image =  Image.open(image_path).convert("RGB")
                img = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
                prev_image_path = image_path
            txt = text_processors["eval"](candidate)
            score = model({"image": img, "text_input": txt}, match_head='itc')
            scores.append(score)

    return scores
