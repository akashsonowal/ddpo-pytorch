
import wandb
import torch 
from fastprogress import progress_bar

from .ppo import compute_loss
from .utils import sd_sample

def sample_and_calculate_rewards(prompts, pipe, image_size, cfg, num_timesteps, decoding_fn, reward_fn, device):
    """
    for a batch
    """
    preds, all_step_preds, log_probs = sd_sample(prompts, pipe, image_size, image_size, cfg, num_timesteps, 1, device)
    imgs = decoding_fn(preds, pipe)    
    rewards = reward_fn(imgs, device)
    return imgs, rewards, all_step_preds, log_probs

def train_one_epoch(args, all_prompts, all_step_preds, log_probs, advantages, pipe, optimizer):

    for inner_epoch in progress_bar(range(args.num_inner_epochs)):
        print(f'Inner epoch {inner_epoch}')

        # chunk them into batches
        all_step_preds_chunked = torch.chunk(all_step_preds, args.num_samples_per_epoch // args.batch_size, dim=1)
        log_probs_chunked = torch.chunk(log_probs, args.num_samples_per_epoch // args.batch_size, dim=1)
        advantages_chunked = torch.chunk(advantages, args.num_samples_per_epoch // args.batch_size, dim=0)
        
        # chunk the prompts (list of strings) into batches
        all_prompts_chunked = [all_prompts[i:i + args.batch_size] for i in range(0, len(all_prompts), args.batch_size)]
        
        for i in progress_bar(range(len(all_step_preds_chunked))):
            optimizer.zero_grad()

            loss = compute_loss(all_step_preds_chunked[i], log_probs_chunked[i], 
                                advantages_chunked[i], args.clip_advantages, args.clip_ratio, all_prompts_chunked[i], pipe, args.num_timesteps, args.cfg, 1, 'cuda'
                                ) # loss.backward happens inside
            
            torch.nn.utils.clip_grad_norm_(pipe.unet.parameters(), 1.0) # gradient clipping
            optimizer.step()
            wandb.log({"loss": loss, "epoch": epoch, "inner_epoch": inner_epoch, "batch": i})
