import numbers
import random
import math
import torch

from torchvision import transforms


class Transforms(object):
    @staticmethod
    def resize(img_size=224):
        # expects a PIL Image
        return transforms.Resize((img_size, img_size))

    @staticmethod
    def affine(degree=5, translate=0.04, scale=0.02):
        # expects a PIL Image
        return transforms.RandomAffine(
            degrees=(-degree, degree),
            translate=(translate, translate),
            scale=(1-scale, 1+scale),
            shear=None)

    @staticmethod
    def random_crop(img_size=224):
        # expects a PIL Image
        return transforms.RandomCrop((img_size, img_size))

    @staticmethod
    def normalize():
        # expects a PIL Image
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    @staticmethod
    def cutout(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0.):
        # expects a tensor
        return transforms.RandomErasing(
            p=p, scale=scale, ratio=ratio, value=value)

    @staticmethod
    def get_transform(transform='default'):
        if transform == 'default':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.normalize()])

        elif transform == 'none':
            return transforms.ToTensor()
        elif transform == 'crops':
            return transforms.Compose([
                Transforms.resize(240),
                Transforms.random_crop(224),
                Transforms.normalize()])
        elif transform == 'cutout':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.normalize(),
                Transforms.cutout()])
        elif transform == 'affine':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.affine(),
                Transforms.normalize()])
        elif transform == 'affine_crops':
            return transforms.Compose([
                Transforms.resize(240),
                Transforms.random_crop(224),
                Transforms.affine(),
                Transforms.normalize()])
        elif transform == 'affine_crops_cutout':
            return transforms.Compose([
                Transforms.resize(240),
                Transforms.random_crop(224),
                Transforms.affine(),
                Transforms.normalize(),
                Transforms.cutout()])
        elif transform == 'affine_cutout':
            return transforms.Compose([
                Transforms.resize(224),
                Transforms.affine(),
                Transforms.normalize(),
                Transforms.cutout()])
        else:
            raise ValueError('Image augmentation {} is not implemented'.format(transform))
