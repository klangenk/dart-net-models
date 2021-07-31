import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

class HungarianMatcher(nn.Module):
    @torch.no_grad()
    def forward(self, pred, targ_label, targ_pos=None):

        bs, num_queries, num_features = pred.shape

        targ_label = targ_label.repeat(1, num_queries // 3)

        # We flatten to compute the cost matrices in a batch
        pred_slice = pred[:, :, :20].softmax(-1).view(bs * num_queries // 3, 3, -1)
        pred_ring = pred[:, :, 20:27].softmax(-1).view(bs * num_queries // 3, 3, -1)

        targ_slice = targ_label.view(bs * num_queries // 3, 3, -1)[:, :, :20].argmax(dim=-1)
        targ_ring = targ_label.view(bs * num_queries // 3, 3, -1)[:, :, 20:27].argmax(dim=-1)

        #cost_pos = 0

        #if targ_pos is not None:
        #    pred_pos = pred[:, :, 27:29].flatten(0, 1)
        #    t_pos = targ_pos.view(bs * num_queries, -1)
        #    cost_pos = torch.cdist(pred_pos, t_pos, p=1)

        mask = targ_ring < 3
        
        #print(pred_slice.shape, pred_ring.shape, targ_slice.shape, targ_ring.shape)
        
        #print(pred_slice[targ_slice].shape, mask.shape, pred_slice.view(bs * num_queries, -1)[targ_slice.view(bs * num_queries)].view(bs, num_queries, -1).shape)
        
        
        cost_slice = -torch.stack([pred_slice[i, :, x] for i, x in enumerate(targ_slice)]) * mask.view(bs * num_queries // 3, 3, 1)
        cost_ring = -torch.stack([pred_ring[i, :, x] for i, x in enumerate(targ_ring)])

        
        # Final cost matrix
        C = cost_slice + cost_ring # + cost_pos
        C = C.view(bs * num_queries // 3, 3, -1).cpu()
        indices = [linear_sum_assignment(c.T) for c in C]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]