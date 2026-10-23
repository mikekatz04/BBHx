# Utilities to aid in bbhx Package Cython GPU/CPU Agnostic setup

# Copyright (C) 2020 Michael L. Katz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Legacy pointer helpers of the Cython CPU/GPU-agnostic framework.

:func:`wrapper` turns ``numpy`` / ``cupy`` arrays (and objects with a ``ptr`` attribute) into
raw integer data pointers, and :func:`pointer_adjust` applies it to every call of a
decorated function. BBHx no longer contains Cython extensions (its native code is bound with
nanobind and takes arrays directly), and nothing in BBHx calls these; the maintained copies
live in ``gpubackendtools.pointeradjust``.

The module-level flag ``gpu`` is ``True`` when ``cupy`` imports.
"""

import os
import subprocess
import warnings

import numpy as np

# check to see if cupy is available for gpus
try:
    import cupy as cp

    gpu = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as np

    gpu = False


# TODO check: keep/remove/document -- legacy Cython helper, no caller; only in docs utils.rst
def wrapper(*args, **kwargs):
    """Convert array and C/C++ class arguments to raw pointers.

    Each argument is checked in turn: a ``cupy.ndarray`` (when ``cupy`` is importable)
    becomes ``arg.data.mem.ptr``, a ``numpy.ndarray`` becomes
    ``arg.__array_interface__["data"][0]``, an object with a ``ptr`` attribute (a Cython
    class) becomes ``arg.ptr``, and anything else is passed through unchanged.

    A Cython consumer must declare the pointer arguments as ``size_t`` and cast them as they
    enter the C++ function; see the Cython codes
    `here <https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/tree/master/src>`_
    for examples.

    Args:
        *args: Positional arguments to convert.
        **kwargs: Keyword arguments to convert (keys kept).

    Returns:
        tuple[list, dict]: ``(targs, tkwargs)``, the converted positional and keyword
        arguments.
    """
    # declare target containers
    targs = []
    tkwargs = {}

    # args first
    for arg in args:
        if gpu:
            # cupy arrays
            if isinstance(arg, cp.ndarray):
                targs.append(arg.data.mem.ptr)
                continue

        # numpy arrays
        if isinstance(arg, np.ndarray):
            targs.append(arg.__array_interface__["data"][0])
            continue

        try:
            # cython classes
            targs.append(arg.ptr)
            continue
        except AttributeError:
            # regular argument
            targs.append(arg)

    # kwargs next
    for key, arg in kwargs.items():
        if gpu:
            # cupy arrays
            if isinstance(arg, cp.ndarray):
                tkwargs[key] = arg.data.mem.ptr
                continue

        if isinstance(arg, np.ndarray):
            # numpy arrays
            tkwargs[key] = arg.__array_interface__["data"][0]
            continue

        try:
            # cython classes
            tkwargs[key] = arg.ptr
            continue
        except AttributeError:
            # other arguments
            tkwargs[key] = arg

    return (targs, tkwargs)


# TODO check: keep/remove/document -- legacy Cython decorator, no caller; only in docs utils.rst
def pointer_adjust(func):
    """Decorate a function so its array arguments arrive as raw pointers.

    The returned function runs :func:`wrapper` on its ``*args`` / ``**kwargs`` and calls
    ``func`` with the converted values. The same ``size_t`` / cast requirement as for
    :func:`wrapper` applies on the Cython side.

    Args:
        func (callable): Function taking pointer arguments (typically a Cython wrapper).

    Returns:
        callable: The wrapping function.
    """

    def func_wrapper(*args, **kwargs):
        # get pointers
        targs, tkwargs = wrapper(*args, **kwargs)
        return func(*targs, **tkwargs)

    return func_wrapper
