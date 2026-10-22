

import torch
import torch.nn as nn
from models.star_plugin import STARPlugin
from layers.StandardNorm import Normalize  

class STARWrapper(nn.Module):
   
    def __init__(self, base_model, configs, global_mask_x=None, global_mask_y=None):
        super(STARWrapper, self).__init__()
        self.base_model = base_model 
        
      
        self.normalization_x = Normalize(configs.enc_in, affine=False)
        
        self.star_plugin = STARPlugin(
            configs, 
            base_encoder=self.base_model.encoder, 
            global_mask_x=global_mask_x, 
            global_mask_y=global_mask_y
        )
        self.pred_len = configs.pred_len

    def forward(self, x_enc, y, is_training):
      
        x_enc_norm = self.normalization_x(x_enc, 'norm')
        B, T, C = x_enc_norm.shape
        enc_out_emb = self.base_model.enc_embedding_x(x_enc_norm, None)
        align_loss = 0.0
        y_res = None
        
     
        if is_training:
            align_loss, y_res = self.star_plugin.compute_align_loss(
                x_enc_norm, y, 
                base_embed_x=self.base_model.enc_embedding_x,       
                base_embed_y=self.base_model.enc_embedding_y,       
                base_enc_layer_0=self.base_model.encoder.attn_layers[0] 
            )
        
      
        enc_out = self.base_model.encoder(enc_out_emb, attn_mask=None)[0]
        
   
        dec_out = self.base_model.projection_x(enc_out).permute(0, 2, 1)[:, :, :C]
        
    
        dec_out_calibrated, pred_exo = self.star_plugin.apply_causal_correction(
            dec_out, y, y_res, is_training
        )
        
      
        final_out = self.normalization_x(dec_out_calibrated, 'denorm')
        pred_exo_denorm = final_out[:, -self.pred_len:, :-1]
        
        return final_out[:, -self.pred_len:, :], y, align_loss, pred_exo_denorm[:, -self.pred_len:, :]