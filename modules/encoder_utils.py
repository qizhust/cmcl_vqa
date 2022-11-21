import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaConfig, RobertaModel
from transformers.models.bert.modeling_bert import BertConfig, BertModel
from modules.clip_model import build_model, adapt_position_encoding
import modules.swin_transformer as swin
from modules.swin_helpers import swin_adapt_position_encoding


class ClipVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = build_model(config.vit, resolution_after=config.image_size)

    def forward(self, data):
        return self.vit(data)

    def fetch_pool_data(self, data):
        return data

    def load_checkpoints(self, state_dict, config):
        state_dict = adapt_position_encoding(state_dict, after=config.image_size, 
                                             patch_size=config.patch_size)
        return state_dict


class SwinVision(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.vit = getattr(swin, config.vit)(pretrained=True, config=config)
        self.avgpool = nn.AdaptiveAvgPool1d(1)

    def forward(self, data):
        return self.vit(x)

    def fetch_pool_data(self, data):
        return F.adaptive_avg_pool1d(data).view(data.size(0), 1, -1)

    def load_checkpoints(self, state_dict, config):
        state_dict = swin_adapt_position_encoding(state_dict, after=config.image_size, 
                                                  before=config.resolution_before)
        return state_dict


def build_vision_encoder(config):
    if 'vit' in config.vit.lower():
        return ClipVision(config)
    elif 'swin' in config.vit.lower():
        return SwinVision(config)
    else:
        raise NotImplementedError

def build_text_encoder(config):
    if 'roberta' in config.tokenizer:
        bert_config_class = RobertaConfig
        bert_model = RobertaModel
    elif 'bert' in config.tokenizer:
        bert_config_class = BertConfig
        bert_model = BertModel
    else:
        raise NotImplementedError

    return bert_config_class(vocab_size=config.vocab_size, 
                             hidden_size=config.hidden_size,
                             num_hidden_layers=config.num_layers,
                             num_attention_heads=config.num_heads,
                             intermediate_size=config.hidden_size * config.mlp_ratio,
                             max_position_embeddings=config.max_text_len,
                             hidden_dropout_prob=config.drop_rate,
                             attention_probs_dropout_prob=config.drop_rate), \
           bert_model.from_pretrained(config.tokenizer) 
