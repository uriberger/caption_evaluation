import clip
import sys
sys.path.append('clipscore')
import clipscore
import torch

def compute_clipscore(candidates, references, image_paths):
    device = torch.device('cuda')
    model, transform = clip.load("ViT-B/32", device=device, jit=False)
    model.eval()
    
    image_feats = clipscore.extract_all_images(image_paths, model, device, batch_size=64, num_workers=8)
    _, per_instance_image_text, candidate_feats = clipscore.get_clip_score(model, image_feats, candidates, device)
    _, per_instance_text_text = clipscore.get_refonlyclipscore(model, references, candidate_feats, device)
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

    return per_instance_image_text, refclipscores
