# SALAD: Skeleton-Aware Latent Diffusion Model for Text-driven Motion Generation and Editing (CVPR 2025)

<p align="center">
    <img src="assets/vae.png"/>
    <br>
    <br>
    <img src="assets/denoiser.png" alt="teaser_image" width="500"/>
    <br>
    <br>
    <img src="assets/editing.png"/>
</p>

<!-- <a href=""><img src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" height=22.5></a> -->
<a href="https://seokhyeonhong.github.io/projects/salad/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=22.5></a>

[Seokhyeon Hong](https://seokhyeonhong.github.io/),
[Chaelin Kim](https://www.linkedin.com/in/chaelin-kim-a942ba218/),
[Serin Yoon](https://serin-yoon.github.io),
[Junghyun Nam](https://vml.kaist.ac.kr/main/people/person/176),
[Sihun Cha](https://https://chacorp.github.io/sihuncha/),
[Junyong Noh](https://vml.kaist.ac.kr/main/people/person/1)

# 1. Preparation
<details>

# Environment
```
conda create -n salad python=3.9 -y
conda activate salad
pip install torch==1.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip install -r requirements.txt
```
We tested our code on ```Python 3.9.19``` and ``PyTorch 1.13.1+cu117``.

Please note that ```requirements.txt``` does not include PyTorch, as its installation depends on your specific hardware and system configuration. To install PyTorch, follow the official installation instructions tailored to your environment, which can be found [here](https://pytorch.org/get-started/previous-versions/).

# Dataset
We used the HumanML3D and KIT-ML datasets, which can be obtained from the following link: [HumanML3D](https://github.com/EricGuo5513/HumanML3D).

After downloading the datasets, please either copy or link them in the following structure:
```
salad
└─ dataset
    └─ humanml3d
    └─ kit-ml
```

# Evaluation & Pre-trained Weights
We provide pre-trained weights for both the HumanML3D and KIT-ML datasets.
To download them, run the following commands:
```
bash prepare/download_t2m.sh
bash prepare/download_kit.sh
```
These scripts will download the pre-trained weights for the SALAD model and evaluation models trained on each dataset.

Additionally, for evaluation, you will need to download the glove as well:
```
bash prepare/download_glove.sh
```

</details>

# 2. Playground
## Generation & Editing
Follow ```playground.ipynb``` to enjoy text-driven motion generation and editing in Jupyter Notebook.

## Attention Map Visualization
Follow ```visualize_attn.ipynb``` if you want to see the attention maps produced during the generation process.

# 3. Training and Evaluation
## Training from Scratch
For training VAE, follow this:
```
python train_vae.py --name vae_example --vae_type vae --lambda_kl 2e-2 --activation silu --dataset_name t2m
```

For training denoiser, follow this:
```
python train_denoiser.py --name denoiser_example --vae_name vae_example --latent_dim 256 --n_heads 8 --ff_dim 1024 --n_layers 5 --num_inference_timesteps 20
```

## Evaluation
```
python test_vae.py --name vae_example
python test_denoiser.py --name denoiser_example --num_inference_timesteps 50
```
Evaluated results will be saved in ```checkpoints/{vae_name | denoiser_name}/eval/eval.log```