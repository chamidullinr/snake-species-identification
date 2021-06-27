import numpy as np
import os

import torch
from fastai.vision.all import Path, DataBlock, ImageBlock, CategoryBlock, ColReader, ColSplitter


def get_valid_col(train_df, label_col, valid_col, *, valid_pct=0.1):
    def label_valid_col(group):
        if group.shape[0] >= 10:
            no_valid_items = int(np.floor(group.shape[0] * valid_pct))
        elif group.shape[0] >= 2:
            no_valid_items = 1
        else:
            no_valid_items = 0

        valid_mask = np.zeros(group.shape[0], dtype=bool)
        if no_valid_items > 0:
            valid_mask[-no_valid_items:] = True

        group[valid_col] = valid_mask
        return group

    return train_df.groupby(label_col).apply(label_valid_col)


def _create_dblock(fn_col, label_col, valid_col, path, folder, item_tfms, batch_tfms,
                   country_col=None):
    pref = f'{Path(path) if folder is None else Path(path)/folder}{os.path.sep}'
    if country_col is None:
        blocks = (ImageBlock, CategoryBlock)
        n_inp = 1
        get_x = ColReader(fn_col, pref=pref, suff='')
        get_y = ColReader(label_col, label_delim=None)
    else:
        blocks = (ImageBlock, CategoryBlock, CategoryBlock)
        n_inp = 2
        get_x = [ColReader(fn_col, pref=pref, suff=''), ColReader(country_col)]
        get_y = ColReader(label_col, label_delim=None)

    dblock = DataBlock(
        blocks=blocks, n_inp=n_inp, get_x=get_x, get_y=get_y,
        splitter=ColSplitter(valid_col), item_tfms=item_tfms, batch_tfms=batch_tfms)
    return dblock


def create_dls(df, *, fn_col, label_col, valid_col, path='.', folder=None,
               bs=64, item_tfms=None, batch_tfms=None,
               num_workers=4, device=torch.device('cpu'), **kwargs):
    dblock = _create_dblock(fn_col, label_col, valid_col, path, folder, item_tfms, batch_tfms)
    dls = dblock.dataloaders(df, path=path, bs=bs, num_workers=num_workers, device=device, **kwargs)
    return dls, dblock


def create_country_dls(df, *, fn_col, country_col, label_col, valid_col, path='.', folder=None,
                       bs=64, item_tfms=None, batch_tfms=None,
                       num_workers=4, device=torch.device('cpu'), **kwargs):
    dblock = _create_dblock(fn_col, label_col, valid_col, path, folder, item_tfms, batch_tfms,
                            country_col)
    dls = dblock.dataloaders(df, path=path, bs=bs, num_workers=num_workers, device=device, **kwargs)
    return dls, dblock
