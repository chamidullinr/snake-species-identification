import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch


def classification_report(preds, targs, *, zero_division=0, **kwargs):
    from sklearn.metrics import classification_report

    report_dict = classification_report(targs, preds, zero_division=zero_division,
                                        output_dict=True, **kwargs)
    return pd.DataFrame(report_dict).T


def plot_confusion_matrix(preds, targs, *, cmap='Blues', **kwargs):
    from sklearn.metrics import confusion_matrix

    labels = np.unique(targs)
    cm = confusion_matrix(targs, preds, labels=labels)
    cm = pd.DataFrame(cm, columns=labels, index=labels)

    fig, ax = plt.subplots(figsize=(16, 10))
    sns.heatmap(cm, cmap=cmap, annot=True, fmt='d', ax=ax)
    plt.show()


def create_eval_df(preds, targs, train_df, *, vocab, label_col, xtra_cols=[]):
    def _merge(df1, df2, on):
        return df1.merge(
            df2.rename(columns=lambda x: f'{x}_{on}' if x != 'idx' else on), 'left', on=on)

    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    if isinstance(targs, torch.Tensor):
        targs = targs.cpu().numpy()

    # create evaluation dataframe
    eval_df = pd.DataFrame(np.array([preds, targs]).T, columns=['preds', 'targs'])

    # include additional label columns
    label_df = train_df[[label_col] + xtra_cols].drop_duplicates().reset_index(drop=True).copy()
    label_df['idx'] = label_df[label_col].replace({item: i for i, item in enumerate(vocab)})
    eval_df = _merge(eval_df, label_df, on='preds')
    eval_df = _merge(eval_df, label_df, on='targs')

    return eval_df
