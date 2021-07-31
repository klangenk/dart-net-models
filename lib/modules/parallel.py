import torch
from fastai.vision.all import *
from fastai.layers import *
from lib.modules.matcher import HungarianMatcher

class MultiDartPlusParallel(torch.nn.Module):
    def __init__(self, arch, encoder=None, head=None, matcher=None, features=81):
        super(MultiDartPlusParallel, self).__init__()
        baseModule = create_body(arch, cut=-2)
        num_features = num_features_model(baseModule)
        pre_head, self.head = self.create_head(num_features * 2, features)
        self.encoder = torch.nn.Sequential(baseModule, pre_head)
        self.matcher = matcher
        if encoder is not None: self.encoder = encoder
        if head is not None: self.head = head


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
            layers.append(torch.nn.Linear(ni, no))
            if actn is not None:
                layers.append(actn)
        if bn_final:
            layers.append(torch.nn.BatchNorm1d(lin_ftrs[-1], momentum=0.01))
        return (torch.nn.Sequential(*layers[:4]), torch.nn.Sequential(*layers[4:]))

    def forward(self, x, darts=None, pos=None):
        batchSize, channels, height, width = x[0].shape
        baseOut = self.encoder(torch.cat(x, 0))
        preHeadOut = baseOut.view(len(x), batchSize, -1).sum(0) / len(x)
        out = self.head(preHeadOut)
        if self.matcher is None: return out
        out = out.view(batchSize, 3, -1)
        indices = self.matcher(out, darts, pos)
        result = torch.cat([out[i][elem[1]] for i, elem in enumerate(indices)])
        return result.view(batchSize, -1)

class MultiDartPlusParallel2(MultiDartPlusParallel):
    def forward(self, x, darts=None, pos=None):
        batchSize, channels, height, width = x[0].shape
        imageCount = len(x)
        baseOut = self.encoder(torch.cat(x, 1).view(batchSize * imageCount, channels, height, width))
        baseOut = baseOut.view(batchSize, imageCount, -1)
        baseOut = torch.cat((baseOut.sum(1, keepdim=True) / imageCount, baseOut), 1)
        baseOut = baseOut.view(batchSize * (imageCount + 1), -1)
        out = self.head(baseOut)
        if self.matcher is None: return out.view(batchSize, -1)
        out = out.view(batchSize, 9, -1)
        indices = self.matcher(out, darts, pos)
        out = out.view(batchSize * 3, 3, -1)
        result = torch.cat([out[i][elem[1]] for i, elem in enumerate(indices)])
        return result.view(batchSize, -1)

class MultiDartPlusParallel3(MultiDartPlusParallel):
    def forward(self, x, darts=None, pos=None):
        batchSize, channels, height, width = x[0].shape
        imageCount = len(x)
        baseOut = self.encoder(torch.cat(x, 1).view(batchSize * imageCount, channels, height, width))
        baseOut = baseOut.view(batchSize, imageCount, -1)
        baseOut = baseOut.sum(1, keepdim=True) / imageCount
        baseOut = baseOut.view(batchSize, -1)
        out = self.head(baseOut)
        if self.matcher is None: return out
        out = out.view(batchSize, 3, -1)
        indices = self.matcher(out, darts, pos)
        result = torch.cat([out[i][elem[1]] for i, elem in enumerate(indices)])
        return result.view(batchSize, -1)


def splitter2(model):
    return [params(model.encoder), params(model.head)]