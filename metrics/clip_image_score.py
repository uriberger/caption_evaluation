from diffusers import AutoPipelineForText2Image
import clip
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

def compute_clip_image_score(candidates, image_paths):
    device = torch.device('cuda')
    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to(device)

    clip_model, preprocess = clip.load("ViT-B/32", device=device)
    cos_sim = nn.CosineSimilarity()

    # Collect references and candidates
    prev_image_path = None
    scores = []
    with torch.no_grad():
        for candidate, image_path in tqdm(list(zip(candidates, image_paths))):
            if image_path != prev_image_path:
                orig_image = Image.open(image_path).convert("RGB")
                orig_image = preprocess(orig_image).unsqueeze(0).to(device)
                orig_image_features = clip_model.encode_image(orig_image)
                prev_image_path = image_path
            reconstructed_image = pipeline_text2image(prompt=candidate).images[0]
            reconstructed_image = preprocess(reconstructed_image).unsqueeze(0).to(device)
            reconstructed_image_features = clip_model.encode_image(reconstructed_image)
            score = cos_sim(orig_image_features, reconstructed_image_features).item()
            scores.append(score)

    return scores
