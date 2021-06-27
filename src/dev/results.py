import pandas as pd
import os

from src.utils import io


def load_specs_files(path):
    """Load specification JSON files and store them as pandas DataFrame."""
    filenames = io.get_filenames_in_dir(path, file_type='json')
    specs = [io.read_json(item) for item in filenames]
    specs_df = pd.DataFrame(specs)

    # sort by date
    assert 'history_file' in specs_df
    specs_df['date'] = specs_df['history_file'].str[-20:-4]
    specs_df = specs_df.sort_values('date').reset_index(drop=True)

    return specs_df


def load_progress_files(specs_df, path):
    """Load training progress CSV files and concatenate them into one pandas DataFrame."""
    assert 'history_file' in specs_df
    out = []
    for _, row in specs_df.iterrows():
        _df = pd.read_csv(os.path.join(path, row['history_file']))
        for col in specs_df.columns:
            if not isinstance(row[col], (list, tuple)):
                _df[col] = row[col]
        out.append(_df)
    out = pd.concat(out, ignore_index=True)
    return out


def filter_items(df, *, outlen=None, copy=False, **kwargs):
    """
    Filter records in pandas DataFrame.

    Filtering is done by **kwargs parameters.
    """
    for k, v in kwargs.items():
        assert k in df, f'Key "{k}" is missing in the dataframe'
        df = df[df[k] == v]
    if outlen is not None:
        assert df.shape[0] == outlen
    if copy:
        df = df.copy()
    return df
