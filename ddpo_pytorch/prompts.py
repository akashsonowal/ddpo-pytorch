import random 

def imagent_animal_prompts():
    animal = random.choice(imagenet_classes[:397])
    prompts = f"{animal}"
    return prompts