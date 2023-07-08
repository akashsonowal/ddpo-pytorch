import argparse 
import wandb

import torch 
from diffusers import StableDiffusionPipeline, DDIMScheduler

import clip 
import requests
from fastprogress import progress_bar, master_bar
import numpy as np
from PIL import Image

from ddpo_pytorch.aesthetic_scorer import MLP, load_aesthetic_model_weights
from ddpo_pytorch.prompts import PromptDataset, imagenet_animal_prompts
from ddpo_pytorch.utils import PerPromptStatTracker, decoding_fn
from ddpo_pytorch.trainer import sample_and_calculate_rewards, train_one_epoch

torch.backends.cuda.matmal.allow_tf32 = True

def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name", default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--enable_attention_slicing", action="store_true")
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true")
    parser.add_argument("--enable_grad_checkpointing", action="store_true")
    parser.add_argument("--num_samples_per_epoch", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--num_inner_epochs", type=int, default=1)
    parser.add_argument("--num_timesteps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--sample_batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_advantages", type=float, default=10.0)
    parser.add_argument("--clip_ratio", type=float, default=1e-4)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--buffer_size", type=int, default=32)
    parser.add_argument("--min_count", type=int, default=16)
    parser.add_argument("--wandb_project", type=str, default="DDPO")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="ddpo_model")
    return parser.parse_args()

def main(args):
    torch.cuda.set_device(args.gpu)

    wandb.init(
        project=args.wandb_project, 
        config={
            "num_samples_per_epoch": args.num_samples_per_epoch,
            "num_epochs": args.num_epochs,
            "num_inner_epochs": args.num_inner_epochs,
            "num_time_steps": args.num_time_steps,
            "batch_size": args.batch_size,
            "lr": args.lr
        }
    )

    pipe = StableDiffusionPipeline.from_pretrained(args.model).to("cuda")

    if args.enable_attention_slicing:
        pipe.enable_attention_slicing()
    
    if args.enable_xformers_memory_efficient_attention:
        pipe.enable_xformers_memory_efficient_attention()
    
    pipe.text_encoder.requires_grad_(False)
    pipe.vae.requires_grad_(False)

    if args.enable_grad_checkpointing: 
        pipe.unet.enable_gradient_checkpointing() # more performance optimization

    pipe.scheduler = DDIMScheduler(
        num_train_timesteps=pipe.scheduler.num_train_timesteps,
        beta_start=pipe.scheduler.beta_start,
        beta_end=pipe.scheduler.beta_end,
        beta_schedule=pipe.scheduler.beta_schedule,
        trained_betas=pipe.scheduler.trained_betas,
        clip_sample=pipe.scheduler.clip_sample,
        set_alpha_to_one=pipe.scheduler.set_alpha_to_one,
        steps_offset=pipe.scheduler.steps_offset,
        prediction_type=pipe.scheduler.prediction_type
    )

    # setup reward model
    clip_model, preprocess = clip.load("ViT-L/14", device="cuda")
    aesthetic_model = MLP(768)
    aesthetic_model.load_state_dict(load_aesthetic_model_weights())
    aesthetic_model.cuda()

    # setup environment
    r = requests.get("https://raw.githubusercontent.com/formigone/tf-imagenet/master/LOC_synset_mapping.txt")
    with open("LOC_synset_mapping.txt", "wb") as f: 
        f.write(r.content)
    
    synsets = {k:v for k,v in [o.split(',')[0].split(' ', maxsplit=1) for o in Path('LOC_synset_mapping.txt').read_text().splitlines()]}
    imagenet_classes = list(synsets.values()) # total 1000 classes

    train_set = PromptDataset(imagenet_animal_prompts, args.num_samples_per_epoch)
    train_dl = torch.utils.data.DataLoader(train_set, batch_size=args.sample_batch_size, shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    per_prompt_stat_tracker = PerPromptStatTracker(args.buffer_size, args.min_count)
  
    def reward_fn(imgs, device):
        clip_model.to(device)
        aesthetic_model.to(device)

        rewards = aesthetic_scoring(imgs, preprocess, clip_model, aesthetic_model_normalize, aesthetic_model)

        clip_model.to("cpu")
        aesthetic_model.to("cpu")
        return rewards

    mean_rewards = [] 

    # start training
    for epoch in master_bar(range(args.num_epochs)):
        print(f"Epoch {epoch}")
        all_step_preds, log_probs, advantages, all_prompts, all_rewards = [], [], [], [], []

        # collect data from environment
        #  sampling `num_samples_per_epoch` images and calculating rewards
        for i, prompts in enumerate(progress_bar(train_dl)):
            batch_imgs, rewards, batch_all_step_preds, batch_log_probs = sample_and_calculate_rewards(prompts, pipe, args.img_size, args.cfg, args.num_timesteps, decoding_fn, reward_fn, 'cuda')
            batch_advantages = torch.from_numpy(per_prompt_stat_tracker.update(np.array(prompts), rewards.squeeze().cpu().detach().numpy())).float().to('cuda')
            wandb.log({"img batch": [wandb.Image(Image.fromarray(img), caption=prompt) for img, prompt in zip(batch_imgs, prompts)]})

            all_step_preds.append(batch_all_step_preds)
            log_probs.append(batch_log_probs)
            advantages.append(batch_advantages)
            all_prompts += prompts
            all_rewards.append(rewards)

        all_step_preds = torch.cat(all_step_preds, dim=1)
        log_probs = torch.cat(log_probs, dim=1)
        advantages = torch.cat(advantages)
        all_rewards = torch.cat(all_rewards)

        mean_rewards.append(all_rewards.mean().item())

        wandb.log({"mean_reward": mean_rewards[-1]})
        wandb.log({"reward_hist": wandb.Histogram(all_rewards.detach().cpu().numpy())})

        # train one epoch
        train_one_epoch(args, all_prompts, all_step_preds, log_probs, advantages, optimizer)
        
        pipe.save_pretrained(args.output_dir)
        wandb.finish()

if __name__ == "__main__":
    args = get_args_parser()
    main(args)
