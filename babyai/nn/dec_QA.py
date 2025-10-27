import os
import pickle as pkl
import torch
from torch import nn
from torch.nn import functional as F

class QAClassifier(nn.Module):
    '''
    object classifier module (a single FF layer)
    '''
    def __init__(self, input_size, vocab_path):
        super().__init__()
        with open(vocab_path, 'rb') as filehandle:
            # read the data as binary data stream
            vocab_list = pkl.load(filehandle)['answer']
        num_classes = len(vocab_list)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out