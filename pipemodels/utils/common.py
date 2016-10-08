import collections

import importlib
import hashlib

import json


def file_checksum(filename, block_size=2**20):
    md5 = hashlib.md5()
    file = open(filename, 'rb')

    while True:
        data = file.read(block_size)
        if not data:
            break
        md5.update(data)

    return md5.hexdigest()


def object_checksum(obj):
    if isinstance(obj, str):
        str_value = obj
    else:
        if isinstance(obj, collections.Mapping):
            obj = dict(obj)

        str_value = json.dumps(obj, sort_keys=True).lower()

    return hashlib.md5(str_value.encode('ascii')).hexdigest()


def split_object_path(object_path):
    split_values = object_path.split('.')
    module, object_name = '.'.join(split_values[:-1]), split_values[-1]
    return module, object_name


def import_object(object_path):
    module, object_name = split_object_path(object_path)
    return getattr(importlib.import_module(module), object_name)
