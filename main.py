import argparse 
import wandb

import torch 
from diffusers import StableDiffusionPipeline, DDIMScheduler

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

if __name__ == "__main__":
    args = get_args_parser()
    main(args)