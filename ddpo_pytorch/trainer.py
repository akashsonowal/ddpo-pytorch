
import wandb
import torch 
from fastprogress import progress_bar

from .ppo import compute_loss
from .utils import sd_sample

def train_one_episode(args, all_prompts, all_step_preds, all_log_probs, all_advantages, pipe, optimizer):

    for epoch in progress_bar(range(args.num_epochs)):
        print(f'Epoch {epoch}')

        # chunk them into batches
        all_step_preds_chunked = torch.chunk(all_step_preds, args.num_samples_per_episode // args.batch_size, dim=1)
        log_probs_chunked = torch.chunk(all_log_probs, args.num_samples_per_episode // args.batch_size, dim=1)
        advantages_chunked = torch.chunk(all_advantages, args.num_samples_per_episode // args.batch_size, dim=0)
        
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
