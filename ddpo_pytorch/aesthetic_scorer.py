import torch
from torch import nn
import requests

class MLP(nn.Module):
    def __init__(self, input_size):
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )
    
    def __repr__(self):
        return """This is the aesthetic model based on Based on https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/fe88a163f4661b4ddabba0751ff645e2e620746e/simple_inference.py."""
    
    def forward(self, x):
        return self.layers(x)

def load_aesthetic_model_weights(cache="."):
    weights_fname = "sac+logos+ava1-l14-linearMSE.pth"
    loadpath = os.path.join(cache, weights_fname)

    if not os.path.exists(loadpath):
        url = {
            "https://github.com/christophschuhmann/"
            f"improved-aesthetic-predictor/blob/main/{weights_fname}?raw=true"
        }
        r = requests.get(url)

        with open(loadpath, "wb") as f:
            f.write(r.content)
    
    weights = torch.load(loadpath, map_location=torch.device("cpu"))
    return weights

def aesthetic_model_normalize(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis)) # calculate l2 norms of each row
    l2[l2 == 0] = 1 # setting 0 norms as 1
    return a / np.expand_dims(l2, axis) # transpose

def aesthetic_scoring(imgs, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model)

