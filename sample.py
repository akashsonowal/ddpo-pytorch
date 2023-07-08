all_rewards = []

# train evaluation
for i, prompts in enumerate(progress_bar(train_dl)):
    batch_imgs, rewards, _, _ = sample_and_calculate_rewards(prompts, pipe, args.img_size, args.cfg, args.num_timesteps, decoding_fn, reward_fn, 'cuda')
    all_rewards.append(rewards)

all_rewards = torch.cat(all_rewards)

mean_rewards = []

mean_rewards.append(all_rewards.mean().item())


