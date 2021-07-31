from fastai.vision.all import *

@delegates()
class MyLoss(BaseLoss):
    @use_kwargs_dict(keep=True, weight=None, ignore_index=-100, reduction='mean')
    def __init__(self, *args, axis=-1, **kwargs): super().__init__(nn.CrossEntropyLoss, *args, axis=axis, **kwargs)    
    def decodes(self, x):
        bs = x.shape[0]
        DARTS = x.shape[1] // 27
        slices = x.view(bs, DARTS, -1)[:, :3, :20].argmax(dim=self.axis)
        rings = x.view(bs, DARTS, -1)[:, :3, 20:27].argmax(dim=self.axis) + 20
        merged = torch.cat((slices.view(-1, 1), rings.view(-1, 1)), 1)
        return torch.cat([one_hot(x, 27) for x in merged], 0).view(-1, 81)
    
    def activation(self, x):
        return x
        bs = x.shape[0]
        DARTS = x.shape[1] // 27
        slices = F.softmax(x.view(bs, DARTS, -1)[:, :, :20], dim=self.axis)
        rings = F.softmax(x.view(bs, DARTS, -1)[:, :, 20:27], dim=self.axis)
        result = torch.cat((slices, rings),-1).view(bs, -1)
        return result

    def __call__(self, pred, targ, **kwargs):
        n, c = targ.shape
        DARTS = pred.shape[1] // 27
        targ = targ.repeat(1, DARTS // 3)
        targ_slice = targ.view(n * DARTS, -1)[:, :20].argmax(dim=-1)
        targ_ring = targ.view(n * DARTS, -1)[:, 20:27].argmax(dim=-1)
        mask = targ_ring < 3
        return super().__call__(
            pred.view(n * DARTS, -1)[mask, :20],
            targ_slice[mask]
        ) + super().__call__(
            pred.view(n * DARTS, -1)[:, 20:27],
            targ_ring
        )