import numpy as np


class SEOBNRv4PHM:
    def __init__(self, max_init_len=-1, use_gpu=False, **kwargs):

        if use_gpu:
            raise NotImplementedError

        else:
            self.xp = np

        if max_init_len > 0:
            self.use_buffers = True
            raise NotImplementedError

        else:
            self.use_buffers = False

        self.allowable_modes = [(2, 2), (3, 3), (4, 4), (2, 1), (3, 2), (4, 3)]

        self.ells_default = self.xp.array([2, 3, 4, 2, 3, 4], dtype=self.xp.int32)

        self.mms_default = self.xp.array([2, 3, 4, 1, 2, 3], dtype=self.xp.int32)

        self.nparams = 2

    def _sanity_check_modes(self, ells, mms):
        for (ell, mm) in zip(ells, mms):
            if (ell, mm) not in self.allowable_modes:
                raise ValueError(
                    "Requested mode [(l,m) = ({},{})] is not available. Allowable modes include {}".format(
                        ell, mm, self.allowable_modes
                    )
                )

    def __call__(
        self,
        m1,
        m2,
        chi1x,
        chi1y,
        chi1z,
        chi2x,
        chi2y,
        chi2z,
        distance,
        phiRef,
        modes=None,
    ):
        if modes is not None:
            ells = self.xp.asarray([ell for ell, mm in modes], dtype=self.xp.int32)
            mms = self.xp.asarray([mm for ell, mm in modes], dtype=self.xp.int32)

            self._sanity_check_modes(ells, mms)

        else:
            ells = self.ells_default
            mms = self.mms_default

        self.num_modes = len(ells)

        self.num_bin_all = len(m1)
        self.lengths = np.asarray(
            [5000 for _ in range(self.num_bin_all)], dtype=np.int32
        )
        self.t = [np.arange(length, dtype=np.float64) * 10.0 for length in self.lengths]
        self.amp_phase = [
            np.asarray(
                [
                    [self.t[i] ** 3 for _ in range(self.num_modes)]
                    for _ in range(self.nparams)
                ]
            ).flatten()
            for i in range(self.num_bin_all)
        ]
