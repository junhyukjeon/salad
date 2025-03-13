import numpy as np
import os
from os.path import join as pjoin
from torch.utils.data import DataLoader

from models.vae.trainer import VAETrainer
from options.vae_option import arg_parse
from utils.fixseed import fixseed

import torch
from utils.get_opt import get_opt
from models.t2m_eval_wrapper import EvaluatorModelWrapper
from motion_loaders.dataset_motion_loader import get_dataset_motion_loader
from models.vae.model import VAE

def load_vae(vae_opt, filename):
    model = VAE(vae_opt)
        
    ckpt = torch.load(pjoin(vae_opt.checkpoints_dir, vae_opt.dataset_name, vae_opt.name, 'model', filename),
                            map_location='cpu')
    model.load_state_dict(ckpt["vae"])
    model.freeze()
    print(f'Loading VAE Model {filename}')
    return model

if __name__ == "__main__":
    opt = arg_parse(is_train=False)
    fixseed(opt.seed)
    
    # evaluation setup
    dataset_opt_path = f"checkpoints/{opt.dataset_name}/Comp_v6_KLD005/opt.txt"
    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
    eval_val_loader, _ = get_dataset_motion_loader(dataset_opt_path, 32, 'test', device=opt.device)

    # evaluation
    vae_opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, "opt.txt")
    vae_opt = get_opt(vae_opt_path, opt.device)

    # model
    vae = load_vae(vae_opt, "net_best_fid.tar").to(opt.device)

    # test
    trainer = VAETrainer(vae_opt, vae)
    trainer.test(eval_wrapper, eval_val_loader, 20,
                 save_dir=pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name, 'eval'), cal_mm=False)