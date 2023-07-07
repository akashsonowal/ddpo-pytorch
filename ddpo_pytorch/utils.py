import torch 

@torch.no_grad()
def sd_sample(prompts, pipe, height, width, guidance_scale, num_inference_steps, eta, device):

    scheduler = pipe.scheduler
    unet = pipe.unet

    text_embeddings = pipe._encode_prompt(prompts, device, 1, do_classifier_free_guidance=guidance_scale > 1.0) # (8, 77, 768)

    scheduler.set_timesteps(num_inference_steps, device=device) # change 1000 to 50
    latents = torch.randn((len(prompts), unet.in_channels, height//8, width//8)).to(device)

    all_step_preds, log_probs = [latents], []


    for i, t in enumerate(progress_bar(scheduler.timesteps)):
        input = torch.cat([latents] * 2)
        input = scheduler.scale_model_input(input, t)

        # predict the noise residual
        pred = unet(input, t, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        pred_uncond, pred_text = pred.chunk(2)
        pred = pred_uncond + guidance_scale * (pred_text - pred_uncond)

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
