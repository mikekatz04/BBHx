import numpy as np
import cupy as cp
from bbhx.utils.constants import *
import time
from scipy.interpolate import CubicSpline

from pyNewSetup import fast_snr_calculator

m1 = 1e6
m2 = 5e6
chi1z = 0.4
chi2z = 0.3
dist = 20.0 * 1e9 * PC_SI
phi_ref = 1.2
f_ref = 0.0
inc = 1.6
lam = 5.4
beta = 1.2
psi = 0.2
t_ref = 0.7 * YRSID_SI
num_binaries = int(1e7)
snr_out = cp.zeros(num_binaries)
params_tmp = cp.tile(cp.array([
    m1, m2, chi1z, chi2z, dist, phi_ref, f_ref, inc, lam, beta, psi, t_ref
]), (num_binaries, 1))

params_tmp[::3, 0] *= 0.98
params = params_tmp.flatten().copy()
N = 256

from bbhx.waveforms.ringdownphenomd import *

# setup splines
spl_ring = CubicSpline(QNMData_a, QNMData_fring)
spl_damp = CubicSpline(QNMData_a, QNMData_fdamp)

# store the coefficients
y_rd = cp.asarray(QNMData_fring).copy()
c1_rd = cp.asarray(spl_ring.c[-2]).copy()
c2_rd = cp.asarray(spl_ring.c[-3]).copy()
c3_rd = cp.asarray(spl_ring.c[-4]).copy()

y_dm = cp.asarray(QNMData_fdamp).copy()
c1_dm = cp.asarray(spl_damp.c[-2]).copy()
c2_dm = cp.asarray(spl_damp.c[-3]).copy()
c3_dm = cp.asarray(spl_damp.c[-4]).copy()

from lisatools.sensitivity import get_sensitivity
from gbgpu.utils.constants import YEAR

df = 1e-6
fmax = 1e-1
freqs = np.arange(0.0, fmax, df)
psd = cp.asarray(get_sensitivity(freqs, sens_fn="noisepsd_AE", model="SciRDv1", includewd=1.0))
num_psd = len(psd)

for i in range(3):
    st = time.perf_counter()
    fast_snr_calculator(
        snr_out, 
        params, 
        num_binaries,
        N,
        y_rd,
        c1_rd,
        c2_rd,
        c3_rd,
        y_dm,
        c1_dm,
        c2_dm,
        c3_dm,
        dspin,
        len(y_rd),
        psd, 
        df, 
        num_psd
    )
    et = time.perf_counter()
    print(i, et - st, num_binaries)
breakpoint()