import torch
import torch.nn as nn
import torch.nn.functional as F



class Filtered_Align(nn.Module):
    def __init__(self, margin=0.05):
        super().__init__()
        self.margin = margin

    def weight_based_dynamic_loss(self, losses):
        n = len(losses)
        w_avg = sum(loss.detach() for loss in losses) / n

        dyn_loss = sum(w_avg * loss / (loss.detach() + 1e-8) for loss in losses)
        
        return dyn_loss

    def forward(self, pred, target):

        pred = nn.functional.normalize(pred, dim=-1)
        target = nn.functional.normalize(target, dim=-1)

        temporal_loss = torch.mean(F.gelu(1.0 - self.margin - F.cosine_similarity(pred, target, dim=-1)))
        
  
        spatial_loss = torch.mean(F.gelu(1.0 - self.margin - F.cosine_similarity(pred, target, dim=-2)))
        

        global_loss = torch.mean(F.gelu(torch.abs(torch.matmul(pred, pred.transpose(1, 2)) - \
                                 torch.matmul(target, target.transpose(1, 2))) - self.margin))

        return self.weight_based_dynamic_loss([temporal_loss, spatial_loss, global_loss])


