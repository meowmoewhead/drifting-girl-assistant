# Adapted from LmeSzinc/AzurLaneAutoScript @ 77f4d01fcd2b0acab05a4d89260e8a0a9cd03d21
# Copyright the ALAS contributors. SPDX-License-Identifier: GPL-3.0-only
from typing import Callable, Generic, TypeVar
T = TypeVar('T')

class cached_property(Generic[T]):
    """
    cached-property from https://github.com/pydanny/cached-property
    Add typing support

    A property that is only computed once per instance and then replaces itself
    with an ordinary attribute. Deleting the attribute resets the property.
    Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
    """

    def __init__(self, func: Callable[..., T]):
        self.func = func

    def __get__(self, obj, cls) -> T:
        if obj is None:
            return self

        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def del_cached_property(obj, name):
    """
    Delete a cached property safely.

    Args:
        obj:
        name (str):
    """
    try:
        del obj.__dict__[name]
    except KeyError:
        pass
