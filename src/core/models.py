import torch.nn as nn
import sys

import timm


class _FunctionWrapper(object):
    def __init__(self, func, func_name):
        self.func = func
        self.func.__name__ = func_name
        self.__name__ = func_name

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __str__(self):
        return self.func.__name__

    def __repr__(self):
        return str(self)


def _create_model_factory(clean_model_name, timm_model_name):
    def create_model(pretrained=False, *args, **kwargs):
        return timm.create_model(timm_model_name, pretrained, *args, **kwargs)

    return _FunctionWrapper(create_model, clean_model_name)


_model_specs = {
    'resnet50': ('tv_resnet50', 224),
    'resnet101': ('tv_resnet101', 224),

    'resnext50': ('resnext50_32x4d', 224),
    'resnext101': ('ig_resnext101_32x8d', 224),

    'resnest26': ('resnest26d', 224),
    'resnest50': ('resnest50d', 224),
    'resnest101': ('resnest101e', 256),
    'resnest200': ('resnest200e', 320),
    'resnest269': ('resnest269e', 416),

    'efficientnet_b0': ('tf_efficientnet_b0', 224),
    'efficientnet_b1': ('tf_efficientnet_b1', 240),
    'efficientnet_b2': ('tf_efficientnet_b2', 260),
    'efficientnet_b3': ('tf_efficientnet_b3', 300),
    'efficientnet_b4': ('tf_efficientnet_b4', 380),
    'efficientnet_b5': ('tf_efficientnet_b5', 456),

    'vit_base_224': ('vit_base_patch16_224', 224),
    'vit_base_384': ('vit_base_patch16_384', 384),
    'vit_large_224': ('vit_large_patch16_224', 224),
    'vit_large_384': ('vit_large_patch16_384', 384)}


# create dictionary with models
MODELS = {k: _create_model_factory(k, v[0]) for k, v in _model_specs.items()}
MODEL_INPUT_SIZES = {k: v[1] for k, v in _model_specs.items()}


# add models as attribute of this module
for k, v in MODELS.items():
    setattr(sys.modules[__name__], k, v)


def get_model_fn(model_arch):
    if callable(model_arch):
        model_fn = model_arch
    elif model_arch in MODELS:
        model_fn = MODELS[model_arch]
    else:
        raise ValueError(f'Unknown model architecture "{model_arch}".')
    return model_fn


def get_model(model_arch, n_out, pretrained=True):
    model_fn = get_model_fn(model_arch)
    model = model_fn(pretrained=pretrained)
    config = model.default_cfg
    in_features = getattr(model, config['classifier']).in_features
    setattr(model, config['classifier'], nn.Linear(in_features, n_out))
    return model


def get_model_config(model_arch):
    model_fn = get_model_fn(model_arch)
    model = model_fn(pretrained=False)
    return model.default_cfg
