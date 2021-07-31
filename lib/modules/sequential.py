import torch
from fastai.vision.all import *

class MultiDartPlus(torch.nn.Module):
    def __init__(self, arch):
        super(MultiDartPlus, self).__init__()
        base_module = create_body(arch, cut=-2)
        pre_head, self.head = self.create_head(1024, 27)
        self.encoder = torch.nn.Sequential(base_module, pre_head)

    def create_head(self, nf, n_out, lin_ftrs=None, ps=0.5, concat_pool=True, bn_final=False, lin_first=False, y_range=None):
        
        "Model head that takes `nf` features, runs through `lin_ftrs`, and out `n_out` classes."
        lin_ftrs = [nf, 512, n_out] if lin_ftrs is None else [nf] + lin_ftrs + [n_out]
        ps = L(ps)
        if len(ps) == 1: ps = [ps[0]/2] * (len(lin_ftrs)-2) + ps
        actns = [torch.nn.ReLU(inplace=True)] * (len(lin_ftrs)-2) + [None]
        pool = AdaptiveConcatPool2d() if concat_pool else torch.nn.AdaptiveAvgPool2d(1)
        layers = [pool, Flatten()]
        for ni, no, p, actn in zip(lin_ftrs[:-1], lin_ftrs[1:], ps, actns):
            layers += [torch.nn.BatchNorm1d(ni)]
            if p != 0:
                layers.append(torch.nn.Dropout(p))
            if len(layers) == 4:
                ni = ni + n_out - 1
            layers.append(torch.nn.Linear(ni, no))
            if actn is not None:
                layers.append(actn)
        if bn_final:
            layers.append(torch.nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
        return (torch.nn.Sequential(*layers[:4]), torch.nn.Sequential(*layers[4:]))

    def forward(self, x, darts):
        imageCount = len(x)
        batchSize, channels, height, width = x[0].shape
        baseOut = self.encoder(torch.cat(x, 0))
        preHeadOut = baseOut.view(imageCount, batchSize, -1).sum(0)
        darts = darts.view(batchSize, 3, -1)[:,:,:26]
        merged = torch.cat([torch.stack((
            torch.cat((preHeadOut[i], d[1] + d[2]), 0),
            torch.cat((preHeadOut[i], d[0] + d[2]), 0),
            torch.cat((preHeadOut[i], d[0] + d[1]), 0)
        ), 0) for i, d in enumerate(darts)], 0)
        return self.head(merged).view(batchSize, -1)

def splitter2(model):
    return [params(model.encoder), params(model.head)]