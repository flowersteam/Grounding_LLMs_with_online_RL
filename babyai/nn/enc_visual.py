import os
import types
import torch
import contextlib
import numpy as np
import torch.nn as nn
import PIL

from PIL import Image
from torchvision import models
from torchvision.transforms import functional as F

from nn.transforms import Transforms

class Resnet18(nn.Module):
    '''
    pretrained Resnet18 from torchvision
    '''
    def __init__(self,
                 device,
                 checkpoint_path=None,
                 share_memory=False):
        super().__init__()
        self.device = device
        self.model = models.resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-3])
        '''if checkpoint_path is not None:
            print('Loading ResNet checkpoint from {}'.format(checkpoint_path))
            model_state_dict = torch.load(checkpoint_path, map_location=device)
            model_state_dict = {
                key: value for key, value in model_state_dict.items()
                if 'GU_' not in key and 'text_pooling' not in key}
            model_state_dict = {
                key: value for key, value in model_state_dict.items()
                if 'fc.' not in key}
            model_state_dict = {
                key.replace('resnet.', ''): value
                for key, value in model_state_dict.items()}
            self.model.load_state_dict(model_state_dict)
        self.model = self.model.to(torch.device(device))'''

        if self.device == 'cuda':
            self.model.cuda()
        self.model = self.model.eval()
        if share_memory:
            self.model.share_memory()
        self._transform = Transforms.get_transform('default')

    def extract(self, x):
        # small image returned by RGBImgPartialObsWrapper transform with resize not necessary
        x = torch.stack([self._transform(Image.fromarray(i.astype('uint8'), 'RGB')).to(torch.device(self.device)) for i in x])
        # x_tensor = torch.tensor(x, dtype=torch.float32)
        return self.model(x)

class FeatureFlat(nn.Module):
    '''
    a few conv layers to flatten features that come out of ResNet
    '''
    def __init__(self, input_shape, output_size):
        super().__init__()
        if input_shape[0] == -1:
            input_shape = input_shape[1:]
        layers, activation_shape = self.init_cnn(
            input_shape, channels=[256, 64], kernels=[1, 1], paddings=[0, 0])
        layers += [
            Flatten(), nn.Linear(np.prod(activation_shape), output_size)]
        self.layers = nn.Sequential(*layers)

    def init_cnn(self, input_shape, channels, kernels, paddings):
        layers = []
        planes_in, spatial = input_shape[0], input_shape[-1]
        for planes_out, kernel, padding in zip(channels, kernels, paddings):
            # do not use striding
            stride = 1
            layers += [
                nn.Conv2d(planes_in, planes_out, kernel_size=kernel,
                          stride=stride, padding=padding),
                nn.BatchNorm2d(planes_out), nn.ReLU(inplace=True)]
            planes_in = planes_out

            spatial = ((spatial - kernel + 2 * padding) // stride) + 1
        activation_shape = (planes_in, spatial, spatial)

        return layers, activation_shape

    def forward(self, frames):
        activation = self.layers(frames)
        return activation


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SimpleEncoder(nn.Module):
    '''
    a simple image encoder that is not pretrained to replace the use of resnet18
    '''
    def __init__(self):
        super().__init__()
        self.image_conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        )
    def forward(self, frame):
        frame_extracted = self.image_conv(frame)
        return frame_extracted
