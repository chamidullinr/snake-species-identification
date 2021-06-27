import json
import os
import re


def to_json(dict_, filename):
    with open(filename, 'w') as f:
        json.dump(dict_, f)


def read_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)


def get_filenames_in_dir(path, file_type=None, include_path=True):
    try:
        files = [os.path.join(path, item) if include_path else item
                 for item in os.listdir(path)]
    except FileNotFoundError:
        files = []

    if file_type is not None:
        files = [item for item in files
                 if re.search(r'(\.{})$'.format(file_type), item)]
    return files
