import torch

from fastai.vision.all import AccumMetric


def accuracy(preds, targs):
    assert preds.ndim == 1 or preds.ndim == 2
    assert targs.ndim == 1

    if preds.ndim == 2:
        preds = preds.argmax(dim=1)
    
    return (preds == targs).float().mean()


def f1_score(preds, targs, *, no_labels=None, labels=None, macro=False, eps=1e-10):
    assert preds.ndim == 1 or preds.ndim == 2
    assert targs.ndim == 1

    if preds.ndim == 2:
        no_labels = preds.shape[1]
        preds = preds.argmax(dim=1)

    # count true positives and false positives and negatives
    if labels is None and no_labels is not None:
        labels = torch.arange(no_labels, device=preds.device)
    elif labels is None:
        labels = torch.unique(torch.cat([targs, preds]))

    labels = labels.reshape(-1, 1)

    # create binary versions of targets and predictions
    targs_bin = (targs == labels).to(targs.dtype)
    preds_bin = (preds == labels).to(preds.dtype)

    # count true positives, false positives and negatives
    tp_sum = (targs_bin * preds_bin).sum(axis=1)  # true positive
    fp_sum = ((1 - targs_bin) * preds_bin).sum(axis=1)  # false positive
    fn_sum = (targs_bin * (1 - preds_bin)).sum(axis=1)  # false negative

    # compute precision and recall
    precision = tp_sum / (tp_sum + fp_sum + eps)
    recall = tp_sum / (tp_sum + fn_sum + eps)

    # compute f1 score
    f1 = 2 * precision * recall / (precision + recall + eps)
    # f1 = f1.clamp(min=eps, max=1-eps)

    # # compute support
    # support = targs_bin.sum(1)

    if macro:
        # precision = torch.mean(precision)
        # recall = torch.mean(recall)
        f1 = torch.mean(f1)
        # support = torch.sum(support)

    # print(f1.item())
    # import fastai
    # print('\t', fastai.vision.all.F1Score(average='macro')(preds, targs))

    return f1


def country_f1_score(preds, targs, country_weights):
    country_weights = country_weights.to(preds.device)

    no_labels = country_weights.shape[0]
    assert torch.all(preds < no_labels), 'Unexpected class (label) in prediction array'
    assert torch.all(targs < no_labels), 'Unexpected class (label) in target array'

    # compute f1 score on species level
    f1 = f1_score(preds, targs, no_labels=no_labels, macro=False)
    assert f1.shape[0] == country_weights.shape[0]

    # compute country f1 score
    f1_country = (f1.reshape(-1, 1) * country_weights).sum(axis=0) / country_weights.sum(axis=0)
    f1_country[torch.isnan(f1_country)] = .0  # replace nans with 0

    # compute macro averaged country f1 score
    macro_f1_country = f1_country.mean()

    return macro_f1_country


def F1Score(no_labels=None, labels=None):
    return AccumMetric(f1_score, dim_argmax=1, no_labels=no_labels, labels=labels, macro=True)


def CountryF1Score(country_weights):
    return AccumMetric(country_f1_score, dim_argmax=1, country_weights=country_weights)
