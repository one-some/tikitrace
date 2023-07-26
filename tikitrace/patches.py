from __future__ import annotations

import inspect
import weakref
from dataclasses import dataclass

import torch

from . import util
from .patch import Patch


@dataclass
class TrackedTensor:
    # Don't keep temporary tensors alive!
    tensor: weakref.ref[torch.Tensor]
    size: int


tracked_tensors = []
tracked_modules = []


def _torch_load(
    patch: Patch,
    f,
    map_location=None,
    pickle_module=None,
    *,
    weights_only=False,
    **pickle_load_args,
):
    patch.log("Loading", f, "to", map_location)

    model = patch.unpatched(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=weights_only,
        **pickle_load_args,
    )
    return model


def _torch_module_init(patch, self, *args, **kwargs):
    tracked_modules.append(weakref.ref(self))
    patch.unpatched(self, *args, **kwargs)


def _torch_tensor_init(patch: Patch, self, *args, **kwargs):
    # Tensor seems to work fine with overwritten __init__...?
    size_bytes = util.get_tensor_size(self)
    tracked_tensors.append(TrackedTensor(weakref.ref(self), size_bytes))

    big_threshold = patch.kwargs["big_threshold_bytes"]
    if big_threshold is not None and size_bytes > big_threshold:
        stack_frame = inspect.stack()[2]
        # Ugly
        pretty_file_name = stack_frame.filename.split("site-packages")[-1].strip("/")
        patch.log(
            f"Big ({util.sizeof_fmt(size_bytes)}) {type(self).__name__} talloc -> {self.device}: {pretty_file_name}:{stack_frame.lineno}"
        )

    if patch.kwargs["log_tensor_init"]:
        patch.log(f"{self.device}\t{util.sizeof_fmt(size_bytes)}\t{self.shape}")


def get_param_name_map():
    param_names = {}
    deepness = {}

    for module_ref in tracked_modules:
        module = module_ref()
        if not module:
            continue

        for d_module, d_rank in deepness.items():
            if module not in d_module.modules():
                continue
            deepness[module] = d_rank + 1
            break

        if not module in deepness:
            deepness[module] = 0

    # Iterate over top level modules
    for big_module in [m for m, rank in deepness.items() if rank == 0]:
        for name, module in big_module.named_modules():
            for p_name, param in module.named_parameters():
                param_names[param] = ".".join([name, p_name])
    return param_names


def _torch_tensor_init_exit(patch: Patch):
    global tracked_tensors
    patch.log("=== Tensor Allocation Overview ===")

    # Remove all dead references
    og_tracked_tensor_count = len(tracked_tensors)
    tracked_tensors = [t for t in tracked_tensors if t.tensor() is not None]

    patch.log(f"Tensors Allocated: {og_tracked_tensor_count}")
    patch.log(f"Living Tensors: {len(tracked_tensors)}")
    patch.log(f"Dead Tensors: {og_tracked_tensor_count - len(tracked_tensors)}")

    patch.log("\n" + ("=" * 35))

    tensors_per_device = {}
    for i in sorted(tracked_tensors, key=lambda x: x.size, reverse=True)[:40]:
        tensor = i.tensor()
        if tensor is None:
            print("! dead !")
            continue

        if tensor.device not in tensors_per_device:
            tensors_per_device[tensor.device] = []
        tensors_per_device[tensor.device].append(i)

    patch.log("=" * 35)
    patch.log("Tensors found per device:")
    total_count = 0
    total_size = 0

    for device, tracked_tensors in tensors_per_device.items():
        total_count += len(tracked_tensors)
        cum_size = sum([t.size for t in tracked_tensors])
        total_size += cum_size
        patch.log(
            f"{device}:\t{len(tracked_tensors)} tensors\t{util.sizeof_fmt(cum_size)}"
        )
    patch.log(f"Total:\t{total_count} tensors\t{util.sizeof_fmt(total_size)}")
    patch.log("=" * 35)

    param_name_map = get_param_name_map()

    for device, tracked_tensors in tensors_per_device.items():
        remaining = len(tracked_tensors)
        print()
        patch.log(f"$$$ Tensor summary for '{device}':")
        for t in sorted(tracked_tensors, key=lambda x: x.size, reverse=True)[:30]:
            tensor = t.tensor()
            name = param_name_map[tensor]

            if tensor.device != device:
                patch.log("DBG: Wrong device for", name, "::", tensor.device, device)

            patch.log(
                f"{name}\t\t{tensor.device}\t{util.sizeof_fmt(i.size)}\t{tensor.shape}"
            )
            remaining -= 1

        if remaining:
            patch.log(f"...and {remaining} others...")
