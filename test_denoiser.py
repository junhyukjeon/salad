import torch
from os.path import join as pjoin
from diffusers import DDIMScheduler

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from models.denoiser.trainer import DenoiserTrainer
from options.denoiser_option import arg_parse

from utils.get_opt import get_opt
from utils.fixseed import fixseed

from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.t2m_eval_wrapper import EvaluatorModelWrapper


def load_vae(vae_opt):
    print(f'Loading VAE Model {vae_opt.name}')

    model = VAE(vae_opt)
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    return model


def load_denoiser(opt, vae_dim):
    print(f'Loading Denoiser Model {opt.name}')
    denoiser = Denoiser(opt, vae_dim)
    ckpt = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'model', 'net_best_fid.tar'),
                            map_location='cpu')
    missing_keys, unexpected_keys = denoiser.load_state_dict(ckpt["denoiser"], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('clip_model.') for k in missing_keys])
    return denoiser


if __name__ == '__main__':
    opt = arg_parse(False)
    vae_name = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt'), opt.device).vae_name
    vae_opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, vae_name, 'opt.txt'), opt.device)

    cond_scale = opt.cond_scale
    num_inference_timesteps = opt.num_inference_timesteps
    opt = get_opt(pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'opt.txt'), opt.device)
    opt.cond_scale = cond_scale
    opt.num_inference_timesteps = num_inference_timesteps
    fixseed(opt.seed)
    
    # evaluation setup
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    # models & noise scheduler
    vae_model = load_vae(vae_opt).to(opt.device)
    denoiser = load_denoiser(opt, vae_opt.latent_dim).to(opt.device)
    scheduler = DDIMScheduler(
        num_train_timesteps=opt.num_train_timesteps,
        beta_start=opt.beta_start,
        beta_end=opt.beta_end,
        beta_schedule=opt.beta_schedule,
        prediction_type=opt.prediction_type,
        clip_sample=False,
    )

    # train
    trainer = DenoiserTrainer(opt, denoiser, vae_model, scheduler)
    trainer.test(eval_wrapper, eval_val_loader, 20,
                 save_dir=pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'eval'), cal_mm=False, save_motion=False)