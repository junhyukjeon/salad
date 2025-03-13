import os
import torch
import torch.nn as nn
# import clip
from os.path import join as pjoin

from transformers import AutoModel, AutoTokenizer
from transformers.utils import move_cache

from transformers import CLIPModel
class FrozenCLIPTextEncoder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, opt):
        super().__init__()
        move_cache()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.opt = opt
        # self.model, _ = clip.load(opt.clip_version, jit=False, device="cpu", download_root=pjoin(opt.checkpoints_dir, "clip"))
        # clip.model.convert_weights(self.model)
        # self.model.to(opt.device)
        if opt.clip_version == "ViT-B/32":
            self.tokenizer = AutoTokenizer.from_pretrained(f"{opt.checkpoints_dir}/clip-vit-base-patch32")
            self.model = AutoModel.from_pretrained(f"{opt.checkpoints_dir}/clip-vit-base-patch32")
        elif opt.clip_version == "ViT-L/14":
            self.tokenizer = AutoTokenizer.from_pretrained(f"{opt.checkpoints_dir}/clip-vit-large-patch14")
            self.model = AutoModel.from_pretrained(f"{opt.checkpoints_dir}/clip-vit-large-patch14")
        else:
            raise ValueError(f"Invalid CLIP version: {opt.clip_version}")
        
        self.max_length = self.tokenizer.model_max_length
        self.freeze()
        print(f"Loaded CLIP text encoder version {opt.clip_version}")

    def freeze(self):
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def encode_text(self, text):
        # text: [B, T]
        # CLIP embedding dimension D
        # tokens = clip.tokenize(text, truncate=True).to(self.opt.device)
        # word_emb = self.model.token_embedding(tokens).type(dtype)
        # word_emb = word_emb + self.model.positional_embedding.type(dtype) # [B, T, D]
        # word_emb = word_emb.permute(1, 0, 2)
        # word_emb = self.model.transformer(word_emb)
        # word_emb = word_emb.permute(1, 0, 2)
        # word_emb = self.model.ln_final(word_emb).type(dtype) # [B, T, D]
        tokens = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        text_input_ids = tokens.input_ids.to(self.model.device)
        text_attn_mask = tokens.attention_mask.to(self.model.device).bool()
        if text_input_ids.shape[-1] > self.max_length:
            text_input_ids = text_input_ids[:, :self.max_length]
        
        word_emb = self.model.text_model(text_input_ids).last_hidden_state

        return word_emb, text_attn_mask, text_input_ids.argmax(dim=-1)
    
    @torch.no_grad()
    def tokenize(self, text):
        tokens = self.tokenizer(text,
                                padding="max_length",
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors="pt")
        return tokens

    @torch.no_grad()
    def decode_text_from_tokens(self, tokens):
        return self.tokenizer.decode(tokens)