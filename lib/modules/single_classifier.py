import torch
from fastai.vision.all import *
from fastai.layers import *

class SingleClassifier(torch.nn.Module):
    def __init__(self, encoder, head):
        super(SingleClassifier, self).__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x):
        batchSize, channels, height, width = x.shape
        return self.head(self.encoder(x))
