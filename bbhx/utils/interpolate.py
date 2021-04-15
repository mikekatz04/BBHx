import numpy as np

try:
    import cupy as xp
    from pyInterpolate import interpolate_wrap as interpolate_wrap_gpu
    from pyInterpolate import interpolate_TD_wrap as interpolate_TD_wrap_gpu

except (ImportError, ModuleNotFoundError) as e:
    print("No CuPy or GPU interpolation available.")
    import numpy as xp

from pyInterpolate_cpu import interpolate_wrap as interpolate_wrap_cpu
from pyInterpolate_cpu import interpolate_TD_wrap as interpolate_TD_wrap_cpu

from bbhx.utils.constants import *


class CubicSplineInterpolant:
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run on GPUs and CPUs.

    args:
        t (1D double xp.ndarray): t values as input for the spline.
        y_all (2D double xp.ndarray): y values for the spline.
            Shape: (ninterps, length).
        use_gpu (bool, optional): If True, prepare arrays for a GPU. Default is
            False.

    """

    def __init__(
        self, x, y_all, length, num_interp_params, num_modes, num_bin_all, use_gpu=False
    ):

        if use_gpu:
            self.xp = xp
            self.interpolate_arrays = interpolate_wrap_gpu

        else:
            self.xp = np
            self.interpolate_arrays = interpolate_wrap_cpu

        ninterps = num_modes * num_interp_params * num_bin_all
        self.degree = 3

        self.length = length

        self.reshape_shape = (num_interp_params, num_bin_all, num_modes, length)

        B = self.xp.zeros((ninterps * length,))
        self.c1 = upper_diag = self.xp.zeros_like(B)
        self.c2 = diag = self.xp.zeros_like(B)
        self.c3 = lower_diag = self.xp.zeros_like(B)
        self.y = y_all

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

        # TODO: need to fix last point
        self.x = x

    @property
    def y_shaped(self):
        return self.y.reshape(self.reshape_shape)

    @property
    def c1_shaped(self):
        return self.c1.reshape(self.reshape_shape)

    @property
    def c2_shaped(self):
        return self.c2.reshape(self.reshape_shape)

    @property
    def c3_shaped(self):
        return self.c3.reshape(self.reshape_shape)

    @property
    def container(self):
        return [self.x, self.y, self.c1, self.c2, self.c3]


class CubicSplineInterpolantTD:
    """GPU-accelerated Multiple Cubic Splines

    This class produces multiple cubic splines on a GPU. It has a CPU option
    as well. The cubic splines are produced with "not-a-knot" boundary
    conditions.

    This class can be run on GPUs and CPUs.

    args:
        t (1D double xp.ndarray): t values as input for the spline.
        y_all (2D double xp.ndarray): y values for the spline.
            Shape: (ninterps, length).
        use_gpu (bool, optional): If True, prepare arrays for a GPU. Default is
            False.

    """

    def __init__(self, x, y_all, lengths, nsubs, num_bin_all, use_gpu=False):

        if use_gpu:
            self.xp = xp
            self.interpolate_arrays = interpolate_TD_wrap_gpu

        else:
            self.xp = np
            self.interpolate_arrays = interpolate_TD_wrap_cpu

        ninterps = nsubs * num_bin_all
        self.degree = 3

        self.lengths = lengths.astype(self.xp.int32)
        self.max_length = lengths.max().item()
        self.num_bin_all = num_bin_all
        self.nsubs = nsubs

        self.reshape_shape = (self.max_length, nsubs, num_bin_all)

        B = self.xp.zeros((ninterps * self.max_length,))
        self.c1 = upper_diag = self.xp.zeros_like(B)
        self.c2 = diag = self.xp.zeros_like(B)
        self.c3 = lower_diag = self.xp.zeros_like(B)
        self.y = y_all

        self.interpolate_arrays(
            x, y_all, B, upper_diag, diag, lower_diag, lengths, num_bin_all, nsubs,
        )
        # TODO: need to fix last point
        self.x = x

    @property
    def t_shaped(self):
        return self.x.reshape(self.max_length, self.num_bin_all).T

    @property
    def y_shaped(self):
        return self.y.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def c1_shaped(self):
        return self.c1.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def c2_shaped(self):
        return self.c2.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def c3_shaped(self):
        return self.c3.reshape(self.max_length, self.nsubs, self.num_bin_all).T

    @property
    def container(self):
        return [self.x, self.y, self.c1, self.c2, self.c3]
