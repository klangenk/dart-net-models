import torch
import torch.nn as nn
from torch_scatter import scatter_mean

class Classifier(nn.Module):
    def __init__(self, encoder, head, scale_enc=2.0):
        super().__init__()
        self.encoder = encoder     # your CNN / ViT / …
        self.head    = head        # your MLP
        self.scale   = scale_enc   # matches your *2/len(indexes)* trick

    def forward(self, images, board_ids, darts):
        """
        images   : (N_images, C, H, W)
        board_ids: (N_images,)  long
        darts    : (N_boards, D_darts)
        """
        # (1) encode every image once
        per_img = self.encoder(images)          # (N_images, F)

        # (2) aggregate by board → mean * 2
        per_board = scatter_mean(per_img, board_ids, dim=0) * self.scale

        # (3) fuse with darts and run head once
        fused = torch.cat([per_board, darts], dim=1)   # (N_boards, F+D_darts)
        return self.head(fused)      