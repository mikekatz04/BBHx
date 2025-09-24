# Interpolant for GPUs

# Copyright (C) 2021 Michael L. Katz
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


import numpy as np

from ..utils.constants import *
from ..utils.parallelbase import BBHxParallelModule

def searchsorted2d_vec(a, b, xp=None, **kwargs):
    if xp is None:
        xp = np

    m, n = a.shape
    max_num = xp.maximum(a.max() - a.min(), b.max() - b.min()) + 1
    r = max_num * xp.arange(a.shape[0])[:, None]
    p = xp.searchsorted((a + r).ravel(), (b + r).ravel(), **kwargs).reshape(m, -1)

    out = p - n * (xp.arange(m)[:, None])
    try:
        xp.cuda.runtime.deviceSynchronize()
    except AttributeError:
        pass

    return out


class CubicSplineInterpolant(BBHxParallelModule):
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines. The cubic splines are produced
    with "not-a-knot" boundary conditions.

    This class has GPU capability.

    Args:
        x (xp.ndarray): f values as input for the spline. Can be 1D flattend array
            of total length
            ``(num_bin_all * length)`` or 2D array with shape: ``(num_bin_all, length)``.
        y_all (xp.ndarray): y values for the spline. This can be a 1D flattened
            array with length
            ``(num_interp_params * num_bin_all * num_modes * length)``
            or 4D arrays of shape: ``(num_interp_params, num_bin_all, num_modes, length)``.
        num_interp_params (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the number of interpolation parameters.
            (Default: ``None``)
        num_bin_all (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the number of total binaries.
            (Default: ``None``)
        num_modes (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the number of modes.
            (Default: ``None``)
        length (int, optional): If ``x`` and ``y_all`` are flattened,
            the user must provide the length of the frequency array for each binary.
            (Default: ``None``)
        force_backend (str, optional): ``"cpu"'', ``"gpu"'', ``"cuda"'', ``"cuda12x"'', or ``"cuda11x"''.

    Raises:
        ValueError: If input arguments are not correct.

    """

    def __init__(
        self,
        x,
        y_all,
        num_interp_params=None,
        num_bin_all=None,
        num_modes=None,
        length=None,
        force_backend=None,
    ):

        # check all inputs

        super().__init__(force_backend=force_backend)

        # first check is for flattened arrays
        if x.ndim == 1 or y_all.ndim == 1:
            if x.ndim != 1 or y_all.ndim != 1:
                raise ValueError(
                    "If providing flattened x and y_all, need to both be flattened."
                )
            if (
                length is None
                or num_modes is None
                or num_bin_all is None
                or num_interp_params is None
            ):
                raise ValueError(
                    "If providing flattened arrays, need to provide dimensional information: length, num_modes, num_bin_all, num_interp_params."
                )

            if len(x) != length * num_modes * num_bin_all:
                raise ValueError(
                    f"Length of the x array is not correct. It is supposed to be {length * num_bin_all}. It is currently {len(x)}."
                )
            if len(y_all) != length * num_modes * num_bin_all * num_interp_params:
                raise ValueError(
                    f"Length of the y_all array is not correct. It is supposed to be {length * num_modes * num_bin_all * num_interp_params}. It is currently {len(y_all)}."
                )

        else:
            # arrays are shaped
            if x.ndim != 3:
                raise ValueError(
                    "If providing shaped array, needs to be 3D with shape (num_bin_all, num_modes, length)."
                )
            if y_all.ndim != 4:
                raise ValueError(
                    "If providing shaped array, needs to be 4D with shape (num_interp_params, num_bin_all, num_modes, length)."
                )

            num_interp_params, num_bin_all, num_modes, length = y_all.shape

            assert y_all.shape[1:] == x.shape

            x = x.flatten()
            y_all = y_all.flatten()

        # get/store info
        ninterps = num_modes * num_interp_params * num_bin_all
        self.degree = 3

        self.length = length

        # get reshape information
        self.reshape_shape = (num_interp_params, num_bin_all, num_modes, length)
        self.x_reshape_shape = (num_bin_all, num_modes, length)

        # setup all arrays for interpolation
        x = self.xp.asarray(x)
        B = self.xp.zeros((ninterps * length,))
        self.c1 = upper_diag = self.xp.zeros_like(B)
        self.c2 = diag = self.xp.zeros_like(B)
        self.c3 = lower_diag = self.xp.zeros_like(B)
        self.y = y_all

        # perform interpolation
        self.interpolate_arrays(
            x,
            y_all,
            B,
            upper_diag,
            diag,
            lower_diag,
            length,
            num_interp_params,
            num_modes,
            num_bin_all,
        )

        self.x = x.copy()

    @property
    def xp(self) -> object:
        """Numpy or Cupy"""
        return self.backend.xp

    @property
    def interpolate_arrays(self) -> callable:
        """C/CUDA wrapped function for computing interpolation."""
        return self.backend.interpolate_wrap

    @property
    def x_shaped(self):
        """Get shaped x array."""
        return self.x.reshape(self.x_reshape_shape)

    @property
    def y_shaped(self):
        """Get shaped y array."""
        return self.y.reshape(self.reshape_shape)

    @property
    def c1_shaped(self):
        """Get shaped c1 array."""
        return self.c1.reshape(self.reshape_shape)

    @property
    def c2_shaped(self):
        """Get shaped c2 array."""
        return self.c2.reshape(self.reshape_shape)

    @property
    def c3_shaped(self):
        """Get shaped c3 array."""
        return self.c3.reshape(self.reshape_shape)

    @property
    def container(self):
        """Container for easy transit of interpolation information."""
        return [self.x, self.y, self.c1, self.c2, self.c3]

    def __call__(self, x_new):
        if x_new.ndim == 3 and x_new.shape[:2] != self.x_reshape_shape[:2]:
            raise ValueError(
                "x_new must have 3D shape of (num_bin_all, num_modes, # of new points)"
            )

        assert self.xp.all(
            x_new <= self.x_shaped.max(axis=-1)[:, :, None]
        ) and self.xp.all(x_new >= self.x_shaped.min(axis=-1)[:, :, None])

        segment_inds = (
            searchsorted2d_vec(
                self.x_shaped.reshape(-1, self.length),
                x_new.reshape(-1, x_new.shape[-1]),
                xp=self.xp,
                side="right",
            )
            - 1
        ).reshape(x_new.shape)

        if self.xp.any(segment_inds == self.length - 1):
            #  assert self.xp.all(x_new[segment_inds == self.length - 1] == self.x_shaped.max(axis=-1))
            segment_inds[segment_inds == self.length - 1] = self.length - 2

        x0 = self.xp.take_along_axis(self.x_shaped, segment_inds, axis=-1)
        y0 = self.xp.take_along_axis(
            self.y_shaped, segment_inds[None, :, :, :], axis=-1
        )
        c1 = self.xp.take_along_axis(
            self.c1_shaped, segment_inds[None, :, :, :], axis=-1
        )
        c2 = self.xp.take_along_axis(
            self.c2_shaped, segment_inds[None, :, :, :], axis=-1
        )
        c3 = self.xp.take_along_axis(
            self.c3_shaped, segment_inds[None, :, :, :], axis=-1
        )

        dx = x_new - x0

        y_new = y0 + c1 * dx + c2 * dx**2 + c3 * dx**3

        if hasattr(self.xp, "cuda"):
            self.xp.get_default_memory_pool().free_all_block()
        return y_new
