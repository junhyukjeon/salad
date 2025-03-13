import argparse
import os
import torch
from os.path import join as pjoin
from utils import paramUtil

def arg_parse(is_train=False):
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    ## basic setup
    parser.add_argument("--name", type=str, default="vae_default", help="Name of this trial")
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU id")

    ## dataloader
    parser.add_argument("--dataset_name", type=str, default="t2m", help="dataset directory", choices=["t2m", "kit"])
    parser.add_argument("--batch_size", default=256, type=int, help="batch size")
    parser.add_argument("--window_size", type=int, default=64, help="training motion length")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers for dataloader")

    ## optimization
    parser.add_argument("--max_epoch", default=50, type=int, help="number of total epochs to run")
    parser.add_argument("--warm_up_iter", default=2000, type=int, help="number of total iterations for warmup")
    parser.add_argument("--lr", default=2e-4, type=float, help="max learning rate")
    parser.add_argument("--milestones", default=[150_000, 250_000], nargs="+", type=int, help="learning rate schedule (iterations)")
    parser.add_argument("--gamma", default=0.05, type=float, help="learning rate decay")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight decay")

    parser.add_argument("--recon_loss", type=str, default="l1_smooth", help="reconstruction loss")
    parser.add_argument("--lambda_recon", type=float, default=1.0, help="reconstruction loss weight")
    parser.add_argument("--lambda_pos", type=float, default=0.5, help="position loss weight")
    parser.add_argument("--lambda_vel", type=float, default=0.5, help="velocity loss weight")
    parser.add_argument("--lambda_kl", type=float, default=0.02, help="kl loss weight") # used when vae

    ## vae arch
    parser.add_argument("--latent_dim", type=int, default=32, help="embedding dimension")
    parser.add_argument("--kernel_size", type=int, default=3, help="kernel size")
    parser.add_argument("--n_layers", type=int, default=2, help="num of layers")
    parser.add_argument("--n_extra_layers", type=int, default=1, help="num of extra layers")
    parser.add_argument("--norm", type=str, default="none", help="normalization", choices=["none", "batch", "layer"])
    parser.add_argument("--activation", type=str, default="gelu", help="activation function", choices=["relu", "silu", "gelu"])
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    
    ## other
    parser.add_argument("--is_continue", action="store_true", help="Name of this trial")
    parser.add_argument("--checkpoints_dir", type=str, default="./checkpoints", help="models are saved here")
    parser.add_argument("--log_every", default=10, type=int, help="iter log frequency")
    parser.add_argument("--save_latest", default=500, type=int, help="iter save latest model frequency")
    parser.add_argument("--eval_every_e", default=1, type=int, help="save eval results every n epoch")

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.meta_dir = pjoin(opt.save_root, 'meta')
    opt.eval_dir = pjoin(opt.save_root, 'animation')
    opt.log_dir = pjoin('./log', opt.dataset_name, opt.name)

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.meta_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    args = vars(opt)

    # non-saved options
    if opt.dataset_name == "t2m":
        opt.data_root = './dataset/humanml3d/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 22
        opt.pose_dim = 263
        opt.contact_joints = [7, 10, 8, 11] # left foot, left toe, right foot, right toe
        opt.fps = 20
        opt.radius = 4
        opt.kinematic_chain = paramUtil.t2m_kinematic_chain
        opt.dataset_opt_path = './checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    elif opt.dataset_name == "kit":
        opt.data_root = './dataset/kit-ml/'
        opt.motion_dir = pjoin(opt.data_root, 'new_joint_vecs')
        opt.text_dir = pjoin(opt.data_root, 'texts')
        opt.joints_num = 21
        opt.pose_dim = 251
        opt.contact_joints = [19, 20, 14, 15]
        opt.fps = 12.5
        opt.radius = 240 * 8
        opt.kinematic_chain = paramUtil.kit_kinematic_chain
        opt.dataset_opt_path = './checkpoints/kit/Comp_v6_KLD005/opt.txt'
    else:
        raise KeyError('Dataset Does not Exists')
    
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    opt.is_train = is_train
    if is_train:
    # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.dataset_name, opt.name)
        if not os.path.exists(expr_dir):
            os.makedirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
    
    return opt