from os.path import join as pjoin
import torch
import numpy as np
from diffusers import DDIMScheduler

from models.vae.model import VAE
from models.denoiser.model import Denoiser
from utils.get_opt import get_opt

def lengths_to_mask(lengths: torch.Tensor) -> torch.Tensor:
    max_frames = torch.max(lengths)
    mask = torch.arange(max_frames, device=lengths.device).expand(
        len(lengths), max_frames) < lengths.unsqueeze(1)
    return mask


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


def cfg_step(denoiser, scheduler, z, timestep, text, cfg_scale=7.5,
             fixed_sa=None, fixed_ta=None, fixed_ca=None):
    pred_uncond, _ = denoiser.forward(z, timestep, [""], need_attn=False)
    pred_cond, (sa, ta, ca) = denoiser.forward(z, timestep, [text], need_attn=True,
                                               fixed_sa=fixed_sa, fixed_ta=fixed_ta, fixed_ca=fixed_ca)
    # z_input = torch.cat([z] * 2, dim=0)
    # text_input = ["", text]
    # pred, (sa, ta, ca) = denoiser.forward(z_input, timestep, text_input, need_attn=True,
    #                                       fixed_sa=fixed_sa, fixed_ta=fixed_ta, fixed_ca=fixed_ca)
    
    # pred_uncond, pred_cond = torch.chunk(pred, 2, dim=0)
    pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
    z = scheduler.step(pred, timestep, z).prev_sample

    return z, (sa, ta, ca)


class Text2Motion:
    """
    Text-to-Motion Generation for a Single Text.
    """
    def __init__(self, denoiser_name, dataset_name="t2m"):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.opt = get_opt(f"checkpoints/{dataset_name}/{denoiser_name}/opt.txt", self.device)
        self.vae_opt = get_opt(f"checkpoints/{dataset_name}/{self.opt.vae_name}/opt.txt", self.device)
        
        self.vae = load_vae(self.vae_opt).to(self.device)
        self.denoiser = load_denoiser(self.opt, self.vae_opt.latent_dim).to(self.device)
        self.scheduler = DDIMScheduler(
            num_train_timesteps=self.opt.num_train_timesteps,
            beta_start=self.opt.beta_start,
            beta_end=self.opt.beta_end,
            beta_schedule=self.opt.beta_schedule,
            prediction_type=self.opt.prediction_type,
            clip_sample=False,
        )
        self.tokenizer = self.denoiser.clip_model.tokenizer
        
        self.vae.eval()
        self.denoiser.eval()
    
    @torch.no_grad()
    def generate(
        self,
        text: str,
        m_lens: int,
        cfg_scale: float=7.5,
        num_inference_timesteps: int=50,
        init_noise: torch.Tensor=None,
    ):
        assert m_lens % 4 == 0, f"m_len should be divisible by 4; got {m_lens}"

        # inputs
        if init_noise is None:
            z = torch.randn(1, m_lens // 4, 7, self.vae_opt.latent_dim).to(self.device, dtype=torch.float32) # 7 for atomic joints
            z = z * self.scheduler.init_noise_sigma
        else:
            z = init_noise
        init_noise = z.clone()
        m_lens = torch.tensor([m_lens]).to(self.device, dtype=torch.float32)

        # set diffusion timesteps
        self.scheduler.set_timesteps(num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(self.device)

        # reverse diffusion
        sa_weights, ta_weights, ca_weights = [], [], []
        for i, timestep in enumerate(timesteps):
            z, (sa, ta, ca) = cfg_step(self.denoiser, self.scheduler, z, timestep, text, cfg_scale=cfg_scale)

            sa_weights.append(sa)
            ta_weights.append(ta)
            ca_weights.append(ca)
        
        # attention weights
        # shape: [bsz, n_timesteps, n_layers, n_heads, n_frames * n_joints, n_words]
        sa_weights = torch.stack(sa_weights, dim=1)
        ta_weights = torch.stack(ta_weights, dim=1)
        ca_weights = torch.stack(ca_weights, dim=1)

        # decode
        motion = self.vae.decode(z)
        if isinstance(motion, tuple) or isinstance(motion, list):
            motion = motion[0]

        return init_noise, motion, (sa_weights, ta_weights, ca_weights)
    
    @torch.no_grad()
    def edit(
        self,
        init_noise: torch.Tensor,
        src_text: str=None,
        edit_text: str=None,
        cfg_scale: float=7.5,
        edit_mode: str="word_swap",
        num_inference_timesteps: int=50,
        src_sa: torch.Tensor=None,
        src_ta: torch.Tensor=None,
        src_ca: torch.Tensor=None,
        src_proportion: float=0.2,
        **kwargs
        # add anything you need or want
    ):
        """
        src_sa: [bsz*nframes, ntimesteps, nlayers, nheads, njoints, njoints]
        src_ta: [bsz*njoints, ntimesteps, nlayers, nheads, nframes, nframes]
        src_ca: [bsz, ntimesteps, nlayers, nheads, nframes*njoints, dclip]
        """
        assert edit_mode in ["word_swap", "refine", "reweight", "mirror"],\
            f"edit_mode should be one of 'word_swap', 'refine', 'reweight'; got {edit_mode}"
        
        # kwargs specific to each edit_mode
        if edit_mode == "mirror":
            assert "mirror_mode" in kwargs, "mirror_mode should be provided for mirror editing."
            assert kwargs["mirror_mode"] in ["lower", "upper", "all"], f"mirror_mode should be one of 'lower', 'upper', 'all'; got {kwargs['mirror_mode']}"
        elif edit_mode == "reweight":
            assert "tgt_word" in kwargs, "tgt_word should be provided for reweight editing."
            assert "reweight_scale" in kwargs, "reweight_scale should be provided for reweight editing."
        elif edit_mode == "word_swap":
            assert "swap_src_proportion" in kwargs, "swap_src_proportion should be provided for word_swap editing."

        # initialize
        z_src = init_noise.clone()
        z_edit = z_src.clone()
        
        # set diffusion timesteps
        self.scheduler.set_timesteps(num_inference_timesteps)
        timesteps = self.scheduler.timesteps.to(self.device)
        
        # reverse diffusion with editing
        for i, timestep in enumerate(timesteps):        
            sa = src_sa[:, i] if src_sa is not None and (i / len(timesteps)) < src_proportion else None
            ta = src_ta[:, i] if src_ta is not None and (i / len(timesteps)) < src_proportion else None

            if edit_mode in ["refine", "word_swap"]:
                _, (edit_sa, edit_ta, edit_ca) = cfg_step(self.denoiser, self.scheduler, z_edit, timestep, edit_text, cfg_scale=cfg_scale)

            if edit_mode == "mirror":
                ca = mirror(src_ca[:, i], kwargs["mirror_mode"])
            elif edit_mode == "reweight":
                ca = reweight(self.tokenizer, src_ca[:, i], src_text, kwargs["tgt_word"], kwargs["reweight_scale"])
            elif edit_mode == "refine":
                ca = refine(self.tokenizer, src_ca[:, i], edit_ca, src_text, edit_text)
            elif edit_mode == "word_swap":
                ca = word_swap(src_ca[:, i], edit_ca, (i / len(timesteps)), kwargs["swap_src_proportion"])
                # ca = word_swap(self.tokenizer, src_text, edit_text)

            z_edit, _ = cfg_step(self.denoiser, self.scheduler, z_edit, timestep, edit_text, cfg_scale=cfg_scale,
                                 fixed_sa=sa, fixed_ta=ta, fixed_ca=ca)
        
        # decode
        edit_motion = self.vae.decode(z_edit)

        return edit_motion


def word_swap(
    src_attn_weights: torch.Tensor=None,
    edit_attn_weights: torch.Tensor=None,
    curr_timestep: float=None,
    src_text_proportion: float=None,
):
    return src_attn_weights if curr_timestep < src_text_proportion else edit_attn_weights


def refine(
    tokenizer,
    src_attn_weights: torch.Tensor,
    edit_attn_weights: torch.Tensor,
    src_text: str,
    edit_text: str,
):
    *_, n_words = src_attn_weights.size()
    x_seq = tokenizer.encode(src_text)
    y_seq = tokenizer.encode(edit_text)
    _, trace_back = global_align(x_seq, y_seq)
    mapper_base = get_aligned_sequences(x_seq, y_seq, trace_back)[-1]
    mapper = torch.zeros(n_words, dtype=torch.int64)
    mapper[:mapper_base.shape[0]] = mapper_base[:, 1]
    mapper[mapper_base.shape[0]:] = len(y_seq) + torch.arange(n_words - len(y_seq))
    mask = (mapper == -1).nonzero(as_tuple=True)[0]

    # print(f"n_words: \t{n_words}")
    # print(f"src_text_len: \t{len(x_seq)}")
    # print(f"edit_text_len: \t{len(y_seq)}")
    # print(f"x_seq: \n{x_seq}")
    # print(f"y_seq: \n{y_seq}")
    # print(f"mapper_base: \n{mapper_base}")
    # print(f"mapper: \n {mapper}")
    # print(mask)
    
    attn_weights = src_attn_weights.clone()
    # attn_weights[..., torch.arange(n_words)] = src_attn_weights[..., mapper]
    attn_weights[..., mask] = edit_attn_weights[..., mask]

    return attn_weights


def reweight(
    tokenizer,
    src_attn_weights: torch.Tensor,
    src_text: str,
    tgt_word: str,
    scale: float,
):
    tgt_idx = get_word_inds(src_text, tgt_word, tokenizer)
    attn_weights = src_attn_weights.clone()
    attn_weights[..., tgt_idx] *= scale
    return attn_weights


def mirror(
    src_attn_weights: torch.Tensor,
    mirror_mode: str="lower",
):
    bsz, n_layers, n_heads, n_frames_n_joints, n_words = src_attn_weights.size()
    n_joints = 7 # atomic joints
    n_frames = n_frames_n_joints // 7

    # reshape
    attn_weights = src_attn_weights.reshape(bsz, n_layers, n_heads, n_frames, n_joints, n_words)

    # swap attn weights
    if mirror_mode == "lower":
        attn_weights[..., (1, 2), :] = attn_weights[..., (2, 1), :]
    elif mirror_mode == "upper":
        attn_weights[..., (4, 5), :] = attn_weights[..., (5, 4), :]
    elif mirror_mode == "all":
        attn_weights[..., (1, 2), :] = attn_weights[..., (2, 1), :]
        attn_weights[..., (4, 5), :] = attn_weights[..., (5, 4), :]

    # reshape back
    attn_weights = attn_weights.reshape(bsz, n_layers, n_heads, n_frames_n_joints, n_words)

    return attn_weights


"""
util functions borrowed from promppt-to-prompt
"""
def get_word_inds(text: str, word_place: int, tokenizer):
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return out

def get_matrix(size_x: int, size_y: int):
    return np.zeros((size_x+1, size_y+1), dtype=np.int32)

def get_traceback_matrix(size_x :int, size_y :int):
    matrix = np.zeros((size_x+1, size_y+1), dtype=np.int32)
    matrix[0, 1:] = 1
    matrix[1:, 0] = 2
    matrix[0, 0] = 4
    return matrix
 
def global_align(x, y):
    matrix = get_matrix(len(x), len(y))
    trace_back = get_traceback_matrix(len(x), len(y))
    
    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            left = matrix[i, j - 1]
            up = matrix[i - 1, j]
            diag = matrix[i - 1, j - 1] + (1 if x[i - 1] == y[j - 1] else -1)
            matrix[i, j] = max(left, up, diag)
            if matrix[i, j] == left:
                trace_back[i, j] = 1
            elif matrix[i, j] == up:
                trace_back[i, j] = 2
            else:
                trace_back[i, j] = 3
    return matrix, trace_back

def get_aligned_sequences(x, y, trace_back: np.ndarray):
    x_seq = []
    y_seq = []
    i = len(x)
    j = len(y)
    mapper_y_to_x = []
    while i > 0 or j > 0:
        if trace_back[i, j] == 3:
            x_seq.append(x[i-1])
            y_seq.append(y[j-1])
            i = i-1
            j = j-1
            mapper_y_to_x.append((j, i))
        elif trace_back[i][j] == 1:
            x_seq.append('-')
            y_seq.append(y[j-1])
            j = j-1
            mapper_y_to_x.append((j, -1))
        elif trace_back[i][j] == 2:
            x_seq.append(x[i-1])
            y_seq.append('-')
            i = i-1
        elif trace_back[i][j] == 4:
            break
    mapper_y_to_x.reverse()
    
    return x_seq, y_seq, torch.tensor(mapper_y_to_x, dtype=torch.int64)

def get_replacement_mapper(x: str, y: str, tokenizer, max_len=77):
    words_x = x.split(' ')
    words_y = y.split(' ')
    if len(words_x) != len(words_y):
        raise ValueError(f"attention replacement edit can only be applied on prompts with the same length"
                         f" but prompt A has {len(words_x)} words and prompt B has {len(words_y)} words.")
    inds_replace = [i for i in range(len(words_y)) if words_y[i] != words_x[i]]
    inds_source = [get_word_inds(x, i, tokenizer) for i in inds_replace]
    inds_target = [get_word_inds(y, i, tokenizer) for i in inds_replace]
    mapper = np.zeros((max_len, max_len))
    i = j = 0
    cur_inds = 0
    while i < max_len and j < max_len:
        if cur_inds < len(inds_source) and inds_source[cur_inds][0] == i:
            inds_source_, inds_target_ = inds_source[cur_inds], inds_target[cur_inds]
            if len(inds_source_) == len(inds_target_):
                mapper[inds_source_, inds_target_] = 1
            else:
                ratio = 1 / len(inds_target_)
                for i_t in inds_target_:
                    mapper[inds_source_, i_t] = ratio
            cur_inds += 1
            i += len(inds_source_)
            j += len(inds_target_)
        elif cur_inds < len(inds_source):
            mapper[i, j] = 1
            i += 1
            j += 1
        else:
            mapper[j, j] = 1
            i += 1
            j += 1

    return torch.from_numpy(mapper).float()
