import math 
import torch 

def calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t):
    std_dev_t = torch.clip(std_dev_t, 1e-6)
    log_probs = -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * std_dev_t ** 2) - torch.log(std_dev_t) - math.log(math.sqrt(2 * math.pi))
    return log_probs

@torch.no_grad()
def sd_sample(prompts, pipe, height, width, guidance_scale, num_inference_steps, eta, device):

    scheduler = pipe.scheduler
    unet = pipe.unet

    text_embeddings = pipe._encode_prompt(prompts, device, 1, do_classifier_free_guidance=guidance_scale > 1.0) # (8, 77, 768) The 8 is because of duplication

    scheduler.set_timesteps(num_inference_steps, device=device) # change 1000 to 50
    latents = torch.randn((len(prompts), unet.in_channels, height//8, width//8)).to(device) # (4, 4, 64, 64)

    all_step_preds, log_probs = [latents], []


    for i, t in enumerate(progress_bar(scheduler.timesteps)):
        input = torch.cat([latents] * 2) # (8, 4, 64, 64)
        input = scheduler.scale_model_input(input, t) # (8, 4, 64, 64)

        # predict the noise residual
        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample # (8, 4, 64, 64) noise

        # perform guidance
        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond) # (4, 4, 64, 64)

        # compute the "previous" noisy sample mean and variance, and get log probs
        scheduler_output = scheduler.step(pred, t, latents, eta, variance_noise=0)
        t_1 = t - scheduler.config.num_train_timesteps // num_inference_steps
        variance = scheduler._get_variance(t, t_1)
        std_dev_t = eta * variance ** (0.5)

        prev_sample_mean = scheduler_output.prev_sample # this is the mean and not full sample since variance is 0
        prev_sample = prev_sample_mean + torch.randn_like(prev_sample_mean) * std_dev_t # get full sample by adding noise

        log_probs.append(calculate_log_probs(prev_sample, prev_sample_mean, std_dev_t).mean(dim=tuple(range(1, prev_sample_mean.ndim))))

        all_step_preds.append(prev_sample)
        latents = prev_sample
    
    return latents, torch.stack(all_step_preds), torch.stack(log_probs)

class PerPromptStatTracker:
    def __init__(self, buffer_size, min_count):
        self.buffer_size = buffer_size # max number of rewards to store for each prompt
        self.min_count = min_count # min number of rewards to store for each prompt 
        self.stats = {}

    def update(self, prompts, rewards):
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards)

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = deque(maxlen=self.buffer_size)
            self.stats[prompt].extend(prompt_rewards)

            if len(self.stats[prompt]) < self.min_count:
                mean = np.mean(rewards)
                std = np.std(rewards) + 1e-6
            else:
                mean = np.mean(self.stats[prompt])
                std = np.std(self.stats[prompt]) + 1e-6
            advantages[prompts == prompt] = (prompt_rewards - mean) / std # advantage is normalized rewards

        return advantages

@torch.no_grad()
def decoding_fn(latents, pipe):
    images = pipe.vae.decode(1 / 0.18215 * latents.cuda()).sample # (4, 3, 512, 512)
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy() # (4, 512, 512, 3)
    images = (images * 255).round().astype("uint8")
    return images 