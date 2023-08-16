import torch.nn as nn
from torchvision.models.swin_transformer import SwinTransformerBlock

class Swin(nn.Module):
    def __init__(self, window_size, shift_size = [0, 0]):
        super(Swin, self).__init__()
        self.encoder = SwinTransformerBlock(dim=256, num_heads=8, window_size=window_size, shift_size=shift_size)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.encoder(x)
        x = x.permute(0, 3, 1, 2)
        return x