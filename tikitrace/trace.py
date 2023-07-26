from contextlib import contextmanager
from typing import Optional

import torch

from . import patches
from .patch import Patch, PatchList


def trace(
    patch_torch_load: bool = True,
    log_tensor_init: bool = False,
    big_threshold_bytes: Optional[int] = None,
):
    """A decorator that patches instrumentation into Torch.

    Args:
        patch_torch_load (bool, optional): Whether to log all calls to `torch.load`. Defaults to True.
        log_tensor_init (bool, optional): Whether to log all tensor initializations. Defaults to False.
        big_threshold_bytes (Optional[int], optional): The amount of bytes a tensor allocation must be to issue a warning,
            or None to disable. Defaults to None.
    """
    def inner(func):
        def decorated_func(*args, **kwargs):
            with trace_ctx(
                patch_torch_load=patch_torch_load,
                log_tensor_init=log_tensor_init,
                big_threshold_bytes=big_threshold_bytes,
            ):
                return func(*args, **kwargs)

        return decorated_func

    return inner


@contextmanager
def trace_ctx(
    patch_torch_load: bool = True,
    log_tensor_init: bool = False,
    big_threshold_bytes: Optional[int] = None,
):
    """A context manager that patches instrumentation into Torch.

    Args:
        patch_torch_load (bool, optional): Whether to log all calls to `torch.load`. Defaults to True.
        log_tensor_init (bool, optional): Whether to log all tensor initializations. Defaults to False.
        big_threshold_bytes (Optional[int], optional): The amount of bytes a tensor allocation must be to issue a warning,
            or None to disable. Defaults to None.
    """
    patch_list = PatchList(
        [
            Patch(torch, "load", patches._torch_load, enable=patch_torch_load),
            Patch(
                torch.Tensor,
                "__init__",
                patches._torch_tensor_init,
                exit_hook=patches._torch_tensor_init_exit,
                context="tensor-init",
                log_tensor_init=log_tensor_init,
                big_threshold_bytes=big_threshold_bytes,
            ),
            Patch(
                torch.nn.Module,
                "__init__",
                patches._torch_module_init,
            ),
        ]
    )

    try:
        with patch_list:
            yield True
    finally:
        pass
