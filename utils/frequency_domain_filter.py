
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
class GlobalMaskCalculator:
    def __init__(self, args, device):
        self.args = args
        self.device = device

    def compute_global_statistics(self, train_loader):
        num_channels = self.args.enc_in
        
      
        first_batch = next(iter(train_loader))
        len_x = first_batch[0].shape[1]
        len_y = self.args.pred_len  


        amp_sum_x = torch.zeros(len_x, num_channels, dtype=torch.float32).to(self.device)
        amp_sq_sum_x = torch.zeros(len_x, num_channels, dtype=torch.float32).to(self.device)
        
        amp_sum_y = torch.zeros(len_y, num_channels, dtype=torch.float32).to(self.device)
        amp_sq_sum_y = torch.zeros(len_y, num_channels, dtype=torch.float32).to(self.device)
        
        count = 0

        with torch.no_grad():
            for data in train_loader:
            
                batch_x = data[0].float().to(self.device)
                amp_x = torch.abs(torch.fft.fft(batch_x, dim=1))
                amp_sum_x += amp_x.sum(dim=0)
                amp_sq_sum_x += (amp_x ** 2).sum(dim=0)
                
             
                batch_y = data[1][:, -self.args.pred_len:, :].float().to(self.device) 
                amp_y = torch.abs(torch.fft.fft(batch_y, dim=1))
                amp_sum_y += amp_y.sum(dim=0)
                amp_sq_sum_y += (amp_y ** 2).sum(dim=0)
                
                count += batch_x.size(0)

      
        mean_x = amp_sum_x / count
        std_x = torch.sqrt((amp_sq_sum_x / count) - (mean_x ** 2) + 1e-5)
        mask_x = mean_x / (std_x + 1e-5)

      
        mean_y = amp_sum_y / count
        std_y = torch.sqrt((amp_sq_sum_y / count) - (mean_y ** 2) + 1e-5)
        mask_y = mean_y / (std_y + 1e-5)

        return mask_x, mask_y

def run_filter(args, train_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    calculator = GlobalMaskCalculator(args, device)
    mask_x, mask_y = calculator.compute_global_statistics(train_loader)
    return mask_x, mask_y

class FrequencyDomainFilter(nn.Module):
    def __init__(self, args, global_mask_amp):
        super(FrequencyDomainFilter, self).__init__()
        self.args = args
        self.register_buffer('mask', global_mask_amp)
    def apply_global_filter(self, data):
        data_fft = torch.fft.fft(data, dim=1)
        mask = self.mask.unsqueeze(0)
        filtered_fft = data_fft * mask
        filtered_data = torch.fft.ifft(filtered_fft, dim=1, n=data.size(1)).real
        return filtered_data
    def forward(self, batch_x):
        return self.apply_global_filter(batch_x)