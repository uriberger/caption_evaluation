import sys
sys.path.append('pacscore')
import evaluation
from evaluation import PACScore, RefPACScore
from models.clip import clip
import torch

def compute_pacscore(candidates, references, image_paths):
        gen = {}
        gts = {}

        ims_cs = list()
        gen_cs = list()
        gts_cs = list()

        if references is None:
            for i, (im_i, gen_i) in enumerate(zip(image_paths, candidates)):
                gen['%d' % (i)] = [gen_i, ]
                gts['%d' % (i)] = gts_i
                ims_cs.append(im_i)
                gen_cs.append(gen_i)
        else:
            for i, (im_i, gts_i, gen_i) in enumerate(zip(image_paths, references, candidates)):
                gen['%d' % (i)] = [gen_i, ]
                gts['%d' % (i)] = gts_i
                ims_cs.append(im_i)
                gen_cs.append(gen_i)
                gts_cs.append(gts_i)

        gts = evaluation.PTBTokenizer.tokenize(gts)
        gen = evaluation.PTBTokenizer.tokenize(gen)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = clip.load('ViT-B/32', device=device)
        model = model.to(device)
        model = model.float()
        checkpoint = torch.load("pacscore/checkpoints/clip_ViT-B-32.pth")
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        _, pac_scores, candidate_feats, len_candidates = PACScore(model, preprocess, ims_cs, gen_cs, device, w=2.0)
        if references is not None:
            _, per_instance_text_text = RefPACScore(model, gts_cs, candidate_feats, device, torch.tensor(len_candidates))
            refpac_scores = 2 * pac_scores * per_instance_text_text / (pac_scores + per_instance_text_text)
        else:
            refpac_scores = None
            
        return pac_scores, refpac_scores
