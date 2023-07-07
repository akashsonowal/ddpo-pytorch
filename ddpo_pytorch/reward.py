import clip 
from .aesthetic_scorer import MLP, load_aesthetic_model_weights, aesthetic_scoring

class RewardModel:
    def __init__(self):
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device="cuda")
        self.aesthetic_model = MLP(768)

        self.aesthetic_model.load_state_dict(load_aesthetic_model_weights())
        self.aesthetic_model.cuda()

    def reward_fn(imgs, device):
        clip_model.to(device)
        aesthetic_model.to(device)

        rewards = aesthetic_scoring(imgs, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model)

        clip_model.to("cpu")
        aesthetic_model.to("cpu")

        return rewards