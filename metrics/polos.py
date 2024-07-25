from polos.models import download_model, load_checkpoint
from PIL import Image

def compute_polos(candidates, references, image_paths):
    images = [Image.open(x).convert("RGB") for x in image_paths]
    polos_data = [{'img': images[i], 'mt': candidates[i], 'refs': references[i]} for i in range(len(images))]

    print('Loading model...', flush=True)
    model_path = download_model("polos")
    model = load_checkpoint(model_path)
    print('Model loaded!')
    print('Computing scores...', flush=True)
    _, scores = model.predict(polos_data, batch_size=8, cuda=True)

    return scores
