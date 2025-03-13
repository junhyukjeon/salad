import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.skeleton.conv import STConv

# def _get_core_adj_matrix():
#     out = torch.zeros(7, 7, dtype=torch.float32)
#     out[0, [1, 2, 3]] = 1
#     out[1, 0] = 1
#     out[2, 0] = 1
#     out[3, [0, 4, 5, 6]] = 1
#     out[4, 3] = 1
#     out[5, 3] = 1
#     out[6, 3] = 1
#     return out

def _get_core_edges():
    return [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5), (3, 6)]

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return x * (scale + 1) + shift


class DenseFiLM(nn.Module):
    def __init__(self, opt):
        super(DenseFiLM, self).__init__()
        self.linear = nn.Sequential(
            nn.SiLU(),
            nn.Linear(opt.latent_dim, opt.latent_dim * 2),
        )

    def forward(self, cond):
        """
        cond: [B, D]
        """
        cond = self.linear(cond)
        cond = cond[:, None, None, :] # unsqueeze for skeleto-temporal dimensions
        scale, shift = cond.chunk(2, dim=-1)
        return scale, shift


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, batch_first=True):
        super(MultiheadAttention, self).__init__()
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.batch_first = batch_first
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, average_attn_weights=False):
        """
        query: [B, T1, D]
        key: [B, T2, D]
        value: [B, T2, D]
        key_padding_mask: [B, T2]
        """
        B, T1, D = query.size()
        _, T2, _ = key.size()

        # linear transformation
        query = self.Wq(query).view(B, T1, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.Wk(key).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            attn_weights = attn_weights.masked_fill(key_padding_mask[:, None, None, :], -1e9)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, value)

        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T1, D)

        # linear transformation
        attn_output = self.Wo(attn_output)

        if need_weights:
            if average_attn_weights:
                attn_weights = attn_weights.mean(dim=1)
            return attn_output, attn_weights
        else:
            return attn_output, None
    
    def forward_with_fixed_attn_weights(self, attn_weights, value):
        """
        Assume that the attention weights are already computed.
        """
        B, H, _, T2 = attn_weights.size()
        D = value.size(-1)

        # linear transformation
        value = self.Wv(value).view(B, T2, self.n_heads, self.head_dim).transpose(1, 2)

        # scaled dot-product attention
        attn_output = torch.matmul(attn_weights, value)

        # concat heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, D)

        # linear transformation
        attn_output = self.Wo(attn_output)

        return attn_output, attn_weights


class STTransformerLayer(nn.Module):
    """
    Setting
        - Normalization first
    """
    def __init__(self, opt):
        super(STTransformerLayer, self).__init__()
        self.opt = opt
        
        # skeletal attention
        self.skel_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.skel_norm = nn.LayerNorm(opt.latent_dim)
        self.skel_dropout = nn.Dropout(opt.dropout)

        # temporal attention
        self.temp_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.temp_norm = nn.LayerNorm(opt.latent_dim)
        self.temp_dropout = nn.Dropout(opt.dropout)

        # cross attention
        self.cross_attn = MultiheadAttention(opt.latent_dim, opt.n_heads, opt.dropout, batch_first=True)
        self.cross_src_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_tgt_norm = nn.LayerNorm(opt.latent_dim)
        self.cross_dropout = nn.Dropout(opt.dropout)

        # ffn
        self.ffn_linear1 = nn.Linear(opt.latent_dim, opt.ff_dim)
        self.ffn_linear2 = nn.Linear(opt.ff_dim, opt.latent_dim)
        self.ffn_norm = nn.LayerNorm(opt.latent_dim)
        self.ffn_dropout = nn.Dropout(opt.dropout)

        # activation
        self.act = F.relu if opt.activation == "relu" else F.gelu

        # FiLM
        self.skel_film = DenseFiLM(opt)
        self.temp_film = DenseFiLM(opt)
        self.cross_film = DenseFiLM(opt)
        self.ffn_film = DenseFiLM(opt)

    def _sa_block(self, x, fixed_attn=None):
        x = self.skel_norm(x)
        if fixed_attn is None:
            x, attn = self.skel_attn.forward(x, x, x, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.skel_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.skel_dropout(x)
        return x, attn

    def _ta_block(self, x, mask=None, fixed_attn=None):
        x = self.temp_norm(x)
        if fixed_attn is None:
            x, attn = self.temp_attn.forward(x, x, x, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.temp_attn.forward_with_fixed_attn_weights(fixed_attn, x)
        x = self.temp_dropout(x)
        return x, attn

    def _ca_block(self, x, mem, mask=None, fixed_attn=None):
        x = self.cross_src_norm(x)
        mem = self.cross_tgt_norm(mem)
        if fixed_attn is None:
            x, attn = self.cross_attn.forward(x, mem, mem, key_padding_mask=mask, need_weights=True, average_attn_weights=False)
        else:
            x, attn = self.cross_attn.forward_with_fixed_attn_weights(fixed_attn, mem)
        x = self.cross_dropout(x)
        return x, attn
    
    def _ff_block(self, x):
        x = self.ffn_norm(x)
        x = self.ffn_linear1(x)
        x = self.act(x)
        x = self.ffn_linear2(x)
        x = self.ffn_dropout(x)
        return x
    
    def forward(self, x, memory, cond, x_mask=None, memory_mask=None,
                skel_attn=None, temp_attn=None, cross_attn=None):

        B, T, J, D = x.size()

        # diffusion timestep embedding
        skel_cond = self.skel_film(cond)
        temp_cond = self.temp_film(cond)
        cross_cond = self.cross_film(cond)
        ffn_cond = self.ffn_film(cond)

        # temporal attention
        ta_out, ta_weight = self._ta_block(x.transpose(1, 2).reshape(B * J, T, D),
                                            mask=x_mask,
                                            fixed_attn=temp_attn)
        ta_out = ta_out.reshape(B, J, T, D).transpose(1, 2)
        ta_out = featurewise_affine(ta_out, temp_cond)
        x = x + ta_out

        # skeletal attention
        sa_out, sa_weight = self._sa_block(x.reshape(B * T, J, D),
                                            fixed_attn=skel_attn)
        sa_out = sa_out.reshape(B, T, J, D)
        sa_out = featurewise_affine(sa_out, skel_cond)
        x = x + sa_out
    
        # cross attention
        ca_out, ca_weight = self._ca_block(x.reshape(B, T * J, D),
                                        memory,
                                        mask=memory_mask,
                                        fixed_attn=cross_attn)
        ca_out = ca_out.reshape(B, T, J, D)
        ca_out = featurewise_affine(ca_out, cross_cond)
        x = x + ca_out

        # feed-forward
        ff_out = self._ff_block(x)
        ff_out = featurewise_affine(ff_out, ffn_cond)
        x = x + ff_out

        attn_weights = (sa_weight, ta_weight, ca_weight)

        return x, attn_weights
    

class SkipTransformer(nn.Module):
    def __init__(self, opt):
        super(SkipTransformer, self).__init__()
        self.opt = opt
        if self.opt.n_layers % 2 != 1:
            raise ValueError(f"n_layers should be odd for SkipTransformer, but got {self.opt.n_layers}")
        
        # transformer encoder
        self.input_blocks = nn.ModuleList()
        self.middle_block = STTransformerLayer(opt)
        self.output_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()

        for i in range((self.opt.n_layers - 1) // 2):
            self.input_blocks.append(STTransformerLayer(opt))
            self.output_blocks.append(STTransformerLayer(opt))
            self.skip_blocks.append(nn.Linear(opt.latent_dim * 2, opt.latent_dim))
        

    def forward(self, x, timestep_emb, word_emb, sa_mask=None, ca_mask=None, need_attn=False,
                fixed_sa=None, fixed_ta=None, fixed_ca=None):
        """
        x: [B, T, J, D]
        timestep_emb: [B, D]
        word_emb: [B, N, D]
        sa_mask: [B, T]
        ca_mask: [B, N]

        fixed_sa: [bsz*nframes, nlayers, nheads, njoints, njoints]
        fixed_ta: [bsz*njoints, nlayers, nheads, nframes, nframes]
        fixed_ca: [bsz, nlayers, nheads, nframes*njoints, dclip]
        """
        # B, T, J, D = x.size()
        
        xs = []

        attn_weights = [[], [], []]
        layer_idx = 0
        for i, block in enumerate(self.input_blocks):
            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca)
            xs.append(x)
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1
        
        sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
        ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
        ca = None if fixed_ca is None else fixed_ca[:, layer_idx]
        x, attns = self.middle_block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                                     skel_attn=sa, temp_attn=ta, cross_attn=ca)
        
        for j in range(len(attn_weights)):
            attn_weights[j].append(attns[j])
        layer_idx += 1

        for (block, skip) in zip(self.output_blocks, self.skip_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = skip(x)

            sa = None if fixed_sa is None else fixed_sa[:, layer_idx]
            ta = None if fixed_ta is None else fixed_ta[:, layer_idx]
            ca = None if fixed_ca is None else fixed_ca[:, layer_idx]

            x, attns = block(x, word_emb, timestep_emb, x_mask=sa_mask, memory_mask=ca_mask,
                             skel_attn=sa, temp_attn=ta, cross_attn=ca)
            
            for j in range(len(attn_weights)):
                attn_weights[j].append(attns[j])
            layer_idx += 1

        if need_attn:
            for j in range(len(attn_weights)):
                attn_weights[j] = torch.stack(attn_weights[j], dim=1)
        else:
            for j in range(len(attn_weights)):
                attn_weights[j] = None

        return x, attn_weights