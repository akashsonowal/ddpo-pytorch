
import clip 
from ddpo_pytorch.aesthetic_scorer import aesthetic_scoring

# setup reward model
clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
aesthetic_model = MLP(768)
aesthetic_model.load_state_dict(load_aesthetic_model_weights())
aesthetic_model.cuda()

def reward_fn(imgs, device):
    clip_model.to(device)
    aesthetic_model.to(device)

    rewards = aesthetic_scoring(imgs, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model)

    clip_model.to("cpu")
    aesthetic_model.to("cpu")
    return rewards