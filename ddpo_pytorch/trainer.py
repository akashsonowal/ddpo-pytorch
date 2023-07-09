
"""
ppo trainer
"""
import wandb
import torch 
from fastprogress import progress_bar
from .utils import sd_sample, calculate_log_probs

def compute_loss(x_t, original_log_probs, advantages, clip_advantages, clip_ratio, prompts, pipe, num_inference_steps, guidance_scale, eta, device):
    """
    clip advantages: 10.0
    clip ratio: 1e-4
    """
    scheduler = pipe.scheduler
    unet = pipe.unet
    text_embeddings = pipe._encode_prompt(prompts,device, 1, do_classifier_free_guidance=guidance_scale > 1.0).detach()
    scheduler.set_timesteps(num_inference_steps, device=device)
    loss_value = 0.

    for i, t in enumerate(progress_bar(scheduler.timesteps)):
        clipped_advantages = torch.clip(advantages, -clip_advantages, clip_advantages).detach()
        
        input = torch.cat([x_t[i].detach()] * 2)
        input = scheduler.scale_model_input(input, t)

        # predict the noise residual
        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

        # compute the "previous" noisy sample mean and variance, and get log probs
        scheduler_output = scheduler.step(pred, t, x_t[i].detach(), eta, variance_noise=0)
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)
        prev_sample_mean = scheduler_output.prev_sample
        current_log_probs = calculate_log_probs(x_t[i+1].detach(), prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim)))

        # calculate loss

        ratio = torch.exp(current_log_probs - original_log_probs[i].detach()) # this is the ratio of the new policy to the old policy

        unclipped_loss = -clipped_advantages * ratio # this is the surrogate loss
        clipped_loss = -clipped_advantages * torch.clip(ratio, 1. - clip_ratio, 1. + clip_ratio) # this is the surrogate loss, but with artificially clipped ratios

        loss = torch.max(unclipped_loss, clipped_loss).mean() # we take the max of the clipped and unclipped surrogate losses, and take the mean over the batch
        loss.backward() 

        loss_value += loss.item()

    return loss_value

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
