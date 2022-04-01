import numpy as np
from torch import Tensor

"""Useful debug watches"""
"""
from pleiotropy.evolution.utils.cantrips import array_shapes
array_shapes
"""
MAX_NUM_RECURSIONS = 8


def array_shapes(loc, ignore_key='__lcl', depth=0):
    ou = {}
    if depth >= MAX_NUM_RECURSIONS:  # max recursion param
        return ou
    for name, value in loc.items():
        if isinstance(value, np.ndarray):
            ou[name] = value.shape
        elif isinstance(value, dict):
            ou[name] = array_shapes(value, depth=depth + 1)
        else:
            try:
                object_ = {key: val for key, val in value.__dict__.items()}
                ou[name] = array_shapes(object_, depth=depth + 1)
            except AttributeError:
                pass

    ou = {key: ou[key] for key in sorted(ou) if ou[key] != {} and key != ignore_key}

    return ou


def tensor_shapes(loc, ignore_key='__lcl', depth=0):
    ou = {}
    if depth >= MAX_NUM_RECURSIONS:
        return ou
    for name, value in loc.items():
        if isinstance(value, Tensor):
            ou[name] = tuple(value.shape)
        elif isinstance(value, dict):
            ou[name] = tensor_shapes(value, depth=depth + 1)
        else:
            try:
                object_ = {key: val for key, val in value.__dict__.items()}
                ou[name] = tensor_shapes(object_, depth=depth + 1)
            except AttributeError:
                pass
    ou = {key: ou[key] for key in sorted(ou) if ou[key] != {} and key != ignore_key}

    rv = {}
    for name, value in ou.items():
        if type(value) == dict:
            for name_nested, value_nested in value.items():
                rv[f"{name}.{name_nested}"] = value_nested
        else:
            rv[name] = value
    return rv


def tensor2numpy(loc, ignore_key='__lcl', depth=0):
    ou = {}
    if depth >= MAX_NUM_RECURSIONS:
        return ou
    for name, value in loc.items():
        if isinstance(value, Tensor):
            ou[name] = value.clone().detach().cpu().numpy()
        elif isinstance(value, dict):
            ou[name] = tensor2numpy(value, depth=depth + 1)
        # elif isinstance(value, list):
        #     ou[name] = tensor2numpy({f"ii-{ii}": value[ii] for ii in range(min(len(value), 4))}, depth=depth+1)
        else:
            try:
                object_ = {key: val for key, val in value.__dict__.items()}
                ou[name] = tensor2numpy(object_, depth=depth + 1)
            except AttributeError:
                pass
    ou = {key: ou[key] for key in sorted(ou) if type(ou[key]) == np.ndarray or (ou[key] != {} and key != ignore_key)}
    # return ou
    rv = {}
    for name, value in ou.items():
        if type(value) == np.ndarray:
            rv[name] = value
        elif type(value) == dict:
            for name_nested, value_nested in value.items():
                rv[f"{name}.{name_nested}"] = value_nested
    return rv
