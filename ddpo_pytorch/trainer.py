from .utils import sd_sample

def sample_and_calculate_rewards(prompts, pipe, image_size, cfg, num_timesteps, decoding_fn, reward_fn, device):
    """
    for a batch
    """
    preds, all_step_preds, log_probs = sd_sample(prompts, pipe, image_size, image_size, cfg, num_timesteps, 1, device)
    imgs = decoding_fn(preds,pipe)    
    rewards = reward_fn(imgs, device)
    return imgs, rewards, all_step_preds, log_probs


