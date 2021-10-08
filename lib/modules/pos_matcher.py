# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn


class HungarianMatcher(nn.Module):
    @torch.no_grad()
    def forward(self, pred, targ_pos, foo=None):
        bs, num_queries, num_features = pred.shape
        mask = targ_pos[:, :, 0] > -1
        cost_pos = torch.cdist(pred.view(bs * 3, -1), targ_pos.view(bs * 3, -1)[mask.view(-1)], p=1)
        # Final cost matrix
        C = cost_pos
        C = C.view(bs, num_queries, -1).cpu()
        sizes = [x.item() for x in torch.sum(mask, -1)]
        indices = [linear_sum_assignment(c[i].T) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
