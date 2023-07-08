import random 
import torch 

class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, prompt_fn, num):
        super().__init__()
        self.prompt_fn = prompt_fn
        self.num = num
        
    def __len__(self): 
        return self.num

    def __getitem__(self, x): 
        return self.prompt_fn()

def imagenet_animal_prompts(imagenet_classes):
    animal = random.choice(imagenet_classes[:397])
    prompts = f"{animal}"
    return prompts
