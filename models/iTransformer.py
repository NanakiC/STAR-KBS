

import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
from layers.StandardNorm import Normalize
class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding_x = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.embed, configs.freq,configs.dropout)
        self.enc_embedding_y = DataEmbedding_inverted(configs.pred_len, configs.d_model, configs.embed, configs.freq,configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        
        # Decoder
        self.projection_x = nn.Linear(configs.d_model, configs.pred_len, bias=True)

        # Normalization
        self.normalization_x = Normalize(configs.enc_in, affine=False)

    def forward(self, x_enc, y, is_training):
        # Normalization from Non-stationary Transformer
        x_enc = self.normalization_x(x_enc, 'norm')
        B, T, C = x_enc.shape
        _, L, _ = y.shape
        # Embedding
        enc_out = self.enc_embedding_x(x_enc, None)
        
        align_loss = 0.0
        # Encoder
        enc_out = self.encoder(enc_out, attn_mask=None)[0]

        
        dec_out = self.projection_x(enc_out).permute(0, 2, 1)[:, :, :C]
        dec_out = self.normalization_x(dec_out, 'denorm')
        
        return dec_out[:, -self.pred_len:, :],y,align_loss
