import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import create_cnn_model, AccumMetric

from . import models


class ModelWrapper():
    """
    Wrapper that functions as a compositor object to CNN object.

    During the forward pass, it takes two inputs (an image and a country variable),
    then processes the image and outputs results of an image and unchanged country variable.
    """

    def __init__(self, cnn_model):
        self.cnn_model = cnn_model

    def parameters(self, *args, **kwargs):
        return self.cnn_model.parameters(*args, **kwargs)

    def children(self, *args, **kwargs):
        return self.cnn_model.children(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.cnn_model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self.cnn_model.load_state_dict(*args, **kwargs)

    def to(self, *args, **kwargs):
        return self.cnn_model.to(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return self.cnn_model.eval(*args, **kwargs)

    def train(self, *args, **kwargs):
        return self.cnn_model.train(*args, **kwargs)

    def forward(self, *x):
        out = self.cnn_model(x[0])
        return out, x[1]

    def __call__(self, *x):
        return self.forward(*x)


def create_model(model_arch, n_out, pretrained=True, use_fastai_cnn=False):
    if use_fastai_cnn:
        model_fn = models.get_model_fn(model_arch)
        model = create_cnn_model(model_fn, n_out, pretrained)
    else:
        model = models.get_model(model_arch, n_out, pretrained)
    model_wrapper = ModelWrapper(model)
    return model_wrapper


def adjust_preds(out, country, country_weights):
    # apply softmax activation on network output
    out = F.softmax(out, dim=1)

    # remove probabilities based on country information
    country_weights = country_weights.to(out.device)
    out = out * country_weights[:, country].T

    # normalize outputs
    out = out / out.sum(1).reshape(-1, 1)

    return out


class CrossEntropyLoss(nn.Module):
    """Cross Entopy loss."""
    def __init__(self, country_weights, apply_adjustment=True):
        super().__init__()
        self.country_weights = country_weights
        self.apply_adjustment = apply_adjustment

        self.eps = 1e-16
        self.reduction = 'mean'
        self.axis = 1

    def forward(self, preds, targs):
        inp, country = preds
        if self.apply_adjustment:
            inp = F.log_softmax(inp, 1)
            inp = inp + torch.log(self.country_weights[:, country].T + self.eps)
            out = F.nll_loss(inp, targs, reduction=self.reduction)
        else:
            out = F.cross_entropy(inp, targs, reduction=self.reduction)
        return out

    def decodes(self, x):
        return x.argmax(dim=self.axis)

    def activation(self, preds):
        if self.apply_adjustment:
            out = adjust_preds(*preds, self.country_weights)
        else:
            out = F.softmax(preds[0], dim=self.axis)
        return out


class CustomAccumMetric(AccumMetric):
    def __init__(self, country_weights, func, dim_argmax, thresh, kwargs):
        super().__init__(func)
        self.country_weights = country_weights
        self.func = func
        self.dim_argmax = dim_argmax
        self.thresh = thresh
        self.kwargs = kwargs
        self.flatten = True
        self.to_np = False
        self.invert_args = False

    def accumulate(self, learner):
        "Store targs and preds from `learn`, using activation function and argmax as appropriate"
        pred = adjust_preds(*learner.pred, self.country_weights)
        if self.dim_argmax:
            pred = pred.argmax(dim=self.dim_argmax)
        if self.thresh:
            pred = (pred >= self.thresh)
        self.accum_values(pred, learner.y, learner)


def _wrap_metric(func, country_weights):
    def wrapper(preds, targs, *args, **kwargs):
        preds = adjust_preds(*preds, country_weights)
        return func(preds, targs, *args, **kwargs)
    return wrapper


def wrap_metric(metric, country_weights):
    if isinstance(metric, AccumMetric):
        out = CustomAccumMetric(
            country_weights, metric.func, metric.dim_argmax,
            metric.thresh, kwargs=metric.kwargs)
    else:
        out = _wrap_metric(metric, country_weights)
    return out


def wrap_metrics(metrics, country_weights):
    return [wrap_metric(met, country_weights) for met in metrics]
