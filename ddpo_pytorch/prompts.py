import os
import random
import requests
from pathlib import Path
import torch


def get_prompts():
    if not os.path.exists("LOC_synset_mapping.txt"):
        r = requests.get(
            "https://raw.githubusercontent.com/formigone/tf-imagenet/master/LOC_synset_mapping.txt"
        )

        with open("LOC_synset_mapping.txt", "wb") as f:
            f.write(r.content)

    synsets = {
        k: v
        for k, v in [
            o.split(",")[0].split(" ", maxsplit=1)
            for o in Path("LOC_synset_mapping.txt").read_text().splitlines()
        ]
    }
    imagenet_classes = list(synsets.values())  # total 1000 classes
    return imagenet_classes


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_fn, num):
        super().__init__()
        self.prompt_fn = prompt_fn
        self.num = num

    def __len__(self):
        return self.num

    def __getitem__(self, x):
        return self.prompt_fn()


def imagenet_animal_prompts():
    imagenet_classes = get_prompts()
    animals = random.choice(imagenet_classes[:397])
    prompts = f"{animals}"
    return prompts