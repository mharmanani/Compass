"""Recursively move nested structures of tensors to a device."""

from __future__ import annotations

import torch


def move_to_device(obj, device, non_blocking: bool = True):
    """Recursively move *obj* to *device*.

    Handles:
    - ``torch.Tensor``
    - ``dict`` (preserves type for ``OrderedDict`` etc.)
    - ``list``
    - ``tuple`` / ``NamedTuple``
    - ``dataclass`` instances

    Non-tensor leaves (``str``, ``int``, ``float``, ``None``, …) are returned
    unchanged.

    Args:
        obj: The object to move.
        device: Target device (e.g. ``"cuda"``, ``torch.device("cpu")``).
        non_blocking: Passed through to ``Tensor.to()``.

    Returns:
        A structure of the same shape with all tensors on *device*.

    Example::

        >>> import torch
        >>> data = {"image": torch.zeros(3, 224, 224), "label": 1}
        >>> data = move_to_device(data, "cuda")
    """

    if isinstance(obj, torch.Tensor):
        return obj.to(device, non_blocking=non_blocking)

    if isinstance(obj, dict):
        return type(obj)({k: move_to_device(v, device, non_blocking) for k, v in obj.items()})

    if isinstance(obj, (list, tuple)):
        items = [move_to_device(v, device, non_blocking) for v in obj]
        # Preserve namedtuple / tuple subclass
        if isinstance(obj, tuple):
            try:
                return type(obj)(*items)  # works for namedtuple
            except TypeError:
                return tuple(items)
        return items

    # dataclass instances
    if hasattr(obj, "__dataclass_fields__"):
        import dataclasses

        changes = {
            f.name: move_to_device(getattr(obj, f.name), device, non_blocking)
            for f in dataclasses.fields(obj)
        }
        return dataclasses.replace(obj, **changes)

    # Everything else (str, int, float, None, np.ndarray, …) — return as-is
    return obj
