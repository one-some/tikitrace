import sys

import torch


def sizeof_fmt(size_bytes):
    # Modified version of https://stackoverflow.com/a/1094933
    for unit in ("Bytes", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if abs(size_bytes) < 1024.0:
            if isinstance(size_bytes, int):
                return f"{size_bytes} {unit}"
            rep = f"{size_bytes:.2f}" if not size_bytes.is_integer() else size_bytes
            return f"{rep} {unit}"
        size_bytes /= 1024.0
    raise ValueError()


def get_tensor_size(tensor: torch.Tensor) -> int:
    return sys.getsizeof(tensor.storage())


def flattened_state_dict(module):
    ret = {}
    for k, v in module.state_dict().items():
        if hasattr(v, "state_dict"):
            ret[k] = flattened_state_dict(v)
        else:
            ret[k] = v
    return ret


def contains_nested(target, big):
    if big == target:
        return True

    for v in big.values():
        if v == target:
            return True
        if isinstance(v, dict) and contains_nested(target, v):
            return True
    print(target, big)
    return False
