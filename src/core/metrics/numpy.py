import numpy as np
import pandas as pd


def p_r_f1_score(preds, targs, *, no_labels=None, labels=None, macro=False, eps=1e-10):
    # count true positives and false positives and negatives
    if labels is None and no_labels is not None:
        labels = np.arange(no_labels)
    elif labels is None:
        labels = np.unique(targs)
    else:
        labels = np.array(labels)

    labels = labels.reshape(-1, 1)

    # create binary versions of targets and predictions
    targs_bin = (targs == labels).astype(targs.dtype)
    preds_bin = (preds == labels).astype(preds.dtype)

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

    # compute support
    support = targs_bin.sum(1)

    if macro:
        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)
        support = np.sum(support)

    return precision, recall, f1, support


def f1_score(preds, targs, *, no_labels=None, labels=None, macro=False, eps=1e-10):
    _, _, f1, _ = p_r_f1_score(preds, targs, no_labels=no_labels, labels=labels,
                               macro=macro, eps=eps)
    return f1


def p_r_f1_report(preds, targs, *, no_labels=None, labels=None, label_names=None):
    if no_labels is None and label_names is not None:
        no_labels = len(label_names)
    if labels is not None and label_names is not None:
        assert len(labels) == len(label_names)

    p, r, f1, s = p_r_f1_score(preds, targs, no_labels=no_labels, labels=labels, macro=False)
    out = pd.DataFrame(
        np.array([p, r, f1, s]).T,
        columns=['precision', 'recall', 'f1-score', 'support'])
    if label_names is not None:
        out.index = label_names

    return out


def test_p_r_f1_score(preds, targs, no_labels):
    from sklearn.metrics import classification_report

    # create classification report with sklearn
    report = classification_report(targs, preds, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).T

    # create custom classification report
    p, r, f1, s = p_r_f1_score(preds, targs, no_labels=no_labels, macro=False)
    custom_report_df = pd.DataFrame(
        np.array([p, r, f1, s]).T, index=np.arange(no_labels),
        columns=['Custom precision', 'Custom recall', 'Custom f1-score', 'Custom support'])

    # merge reports and test equality of each class
    _report_df = report_df.iloc[:-3].copy()
    _report_df.index = _report_df.index.astype(int)
    report_df_2 = pd.concat([_report_df, custom_report_df], axis=1).sort_index()
    report_df_2 = report_df_2[report_df_2['Custom support'] != 0]

    for col in ['precision', 'recall', 'f1-score', 'support']:
        assert np.allclose(report_df_2[col], report_df_2['Custom ' + col])

    # test equality of macro metrics
    mp, mr, mf1, ms = p_r_f1_score(preds, targs, labels=_report_df.index.values, macro=True)
    assert np.allclose(report_df.loc['macro avg'], (mp, mr, mf1, ms))


def clean_country_map(country_map_df, species, countries=None, *, missing_val=0, verbose=0):
    from pprint import pprint

    out = country_map_df.copy()

    # add misssing species
    missing_species = {i: item for i, item in enumerate(species) if item not in out.index}
    if verbose:
        print('Missing Species:')
        pprint(missing_species)

    for item in missing_species.values():
        out.loc[item] = missing_val

    if countries is not None:
        # add missing countries
        missing_countries = {i: item for i, item in enumerate(countries) if item not in out.columns}
        if verbose:
            print('\nMissing Countries:')
            pprint(missing_countries)

        for item in missing_countries.values():
            out[item] = missing_val

    # remove duplicate countries
    dup = out.columns.duplicated()
    for col in out.columns[dup]:
        series = out[col].apply(lambda x: np.sign(np.sum(x)), axis=1)
        out.drop(columns=[col], inplace=True)
        out[col] = series

    if countries is not None:
        # keep only needed species and countries in correct order
        out = out.loc[species, countries]
        assert out.shape == (len(species), len(countries)), out.shape
    else:
        # keep only needed species in correct order
        out = out.loc[species]
        assert out.shape[0] == len(species), out.shape

    return out


def country_f1_score(preds, targs, country_weights):
    no_labels = country_weights.shape[0]
    assert np.all(preds < no_labels), 'Unexpected class (label) in prediction array'
    assert np.all(targs < no_labels), 'Unexpected class (label) in target array'

    # compute f1 score on species level
    _, _, f1, _ = p_r_f1_score(preds, targs, no_labels=no_labels, macro=False)
    assert f1.shape[0] == country_weights.shape[0]

    # compute country f1 score
    f1_country = (f1.reshape(-1, 1) * country_weights).sum(axis=0) / country_weights.sum(axis=0)
    f1_country[np.isnan(f1_country)] = .0  # replace nans with 0

    # compute macro averaged country f1 score
    macro_f1_country = f1_country.mean()

    return macro_f1_country
