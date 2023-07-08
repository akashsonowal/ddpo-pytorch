# RLHF for Diffusion Models

Citing: This codebase is clone of [work by Tanishq Abraham](https://github.com/tmabraham/ddpo-pytorch/tree/main). I have cloned for my own learning purpose but it may be useful if you are looking for little more organized code than the original repo. I have tried to keep the notation little more beginner friendly.

This is an implementation of [Training Diffusion Models with Reinforcement Learning](https://arxiv.org/abs/2305.13301) with only basic features. It currently only implements [LAION aesthetic classifier](https://github.com/LAION-AI/aesthetic-predictor) as a reward function.

## Installation
```
git clone https://github.com/akashsonowal/ddpo-pytorch.git 
cd ddpo-pytorch
pip install -r requirements.txt
```

## Usage

It's as simple as running:
```
python main.py
```

To save memory (you'll likely need it), use the arguments `--enable_attention_slicing`, `--enable_xformers_memory_efficient_attention`, and `--enable_grad_checkpointing`. 

## Results

Original samples:
![image](https://github.com/tmabraham/ddpo-pytorch/assets/37097934/6a9489a2-9cfb-4e21-84c5-eaa2694acbd4)

After training for 50 epochs:
![image](https://github.com/tmabraham/ddpo-pytorch/assets/37097934/a82ce5ce-2e29-4adf-b06c-601295be288d)
