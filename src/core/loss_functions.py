import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import CrossEntropyLossFlat, LabelSmoothingCrossEntropyFlat


class F1Loss(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

        self.reduction = 'mean'

    def f1_loss(self, preds, targs, macro=True):
        assert preds.ndim == 1 or preds.ndim == 2
        assert targs.ndim == 1

        if preds.ndim == 2:
            preds = F.softmax(preds, dim=1)

        # count true positives and false positives and negatives
        labels = torch.arange(preds.shape[1], device=preds.device).reshape(-1, 1)

        # # create binary versions of targets and predictions
        targs_bin = (targs == labels).to(targs.dtype)
        preds = preds.T

        # count true positives, false positives and negatives
        tp_sum = (targs_bin * preds).sum(axis=1)  # true positive
        fp_sum = ((1 - targs_bin) * preds).sum(axis=1)  # false positive
        fn_sum = (targs_bin * (1 - preds)).sum(axis=1)  # false negative

        # compute precision and recall
        precision = tp_sum / (tp_sum + fp_sum + self.eps)
        recall = tp_sum / (tp_sum + fn_sum + self.eps)

        # compute f1 loss
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        f1 = f1.clamp(min=self.eps, max=1 - self.eps)

        if macro:
            f1 = torch.mean(f1)

        return 1 - f1

    def forward(self, preds, targs):
        return self.f1_loss(preds, targs)

    def decodes(self, x):
        return x.argmax(dim=1)

    def activation(self, x):
        return F.softmax(x, dim=1)


def ce_loss(*args, weight=None, **kwargs):
    return CrossEntropyLossFlat(weight=weight)


def ce_weighted_loss(country_weights, *args, **kwargs):
    def laplace_smoothing(freq):
        # bayesian approach
        probs = (freq + 1) / (freq.sum() + len(freq))
        assert abs(probs.sum() - 1) < 1e-16
        return probs

    priors = laplace_smoothing(freq=country_weights.sum(1))

    return CrossEntropyLossFlat(weight=priors)


def lsce_loss(*args, eps=0.1, weight=None, **kwargs):
    return LabelSmoothingCrossEntropyFlat(eps=eps, weight=weight)


def f1_loss(*args, **kwargs):
    return F1Loss()


LOSSES = {
    'ce': ce_loss,
    'ce_weighted': ce_weighted_loss,
    'lsce': lsce_loss,
    'f1': f1_loss
}
