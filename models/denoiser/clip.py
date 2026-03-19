import os
import torch
import torch.nn as nn

from transformers import AutoTokenizer, CLIPModel
from transformers.utils import move_cache


class FrozenCLIPTextEncoder(nn.Module):
    """
    Uses the CLIP transformer encoder for text.
    """
    def __init__(self, opt):
        super().__init__()
        move_cache()
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        self.opt = opt

        if opt.clip_version == "ViT-B/32":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="./checkpoints/clip-vit-base-patch32",
            )
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir="./checkpoints/clip-vit-base-patch32",
            )
        elif opt.clip_version == "ViT-L/14":
            self.tokenizer = AutoTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir="./checkpoints/clip-vit-large-patch14",
            )
            self.model = CLIPModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                cache_dir="./checkpoints/clip-vit-large-patch14",
            )
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
        """
        Token-level text embeddings for cross-attention.
        Returns:
            word_emb: [B, 77, D]
            text_attn_mask: [B, 77] with True for valid tokens
            eos_pos: [B]
        """
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        text_input_ids = tokens.input_ids.to(self.model.device)
        text_attn_mask = tokens.attention_mask.to(self.model.device).bool()

        if text_input_ids.shape[-1] > self.max_length:
            text_input_ids = text_input_ids[:, :self.max_length]
            text_attn_mask = text_attn_mask[:, :self.max_length]

        word_emb = self.model.text_model(
            input_ids=text_input_ids,
            attention_mask=text_attn_mask,
        ).last_hidden_state

        return word_emb, text_attn_mask, text_input_ids.argmax(dim=-1)

    @torch.no_grad()
    def encode_text_pooled(self, text, normalize=True):
        """
        CLIP-native pooled text feature for sentence/style similarity.
        Returns:
            text_features: [B, D_proj]
        """
        tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        text_input_ids = tokens.input_ids.to(self.model.device)
        text_attn_mask = tokens.attention_mask.to(self.model.device)

        if text_input_ids.shape[-1] > self.max_length:
            text_input_ids = text_input_ids[:, :self.max_length]
            text_attn_mask = text_attn_mask[:, :self.max_length]

        text_features = self.model.get_text_features(
            input_ids=text_input_ids,
            attention_mask=text_attn_mask,
        )

        if normalize:
            text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-8)

        return text_features

    @torch.no_grad()
    def tokenize(self, text):
        return self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    @torch.no_grad()
    def decode_text_from_tokens(self, tokens):
        return self.tokenizer.decode(tokens)