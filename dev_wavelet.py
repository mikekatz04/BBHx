import numpy as np
from bbhx.waveformbuild import BBHWaveformFD
from lisatools.utils.constants import *
from lisatools.detector import EqualArmlengthOrbits
import matplotlib.pyplot as plt
from bbhx.utils.interpolate import CubicSplineInterpolant


WAVELET_BANDWIDTH = 6.51041666666667e-5
WAVELET_DURATION = 7680.0
LISA_CADENCE = 5

from dataclasses import dataclass


@dataclass
class WDMSetup:

    def __init__(
        self,
        Tobs: float, 
        oversample: int = 1,
    ):
        self.NT = int(np.ceil(Tobs/WAVELET_DURATION).astype(int))
        self.NF = int(WAVELET_DURATION/LISA_CADENCE)
        self.df = WAVELET_BANDWIDTH
        self.dt = WAVELET_DURATION
        self.oversample = oversample

        self.cadence = WAVELET_DURATION/self.NF
        self.Omega = np.pi/self.cadence
        self.dOmega = self.Omega/self.NF
        self.inv_root_dOmega = 1.0/np.sqrt(self.dOmega)
        self.B = self.dOmega
        self.A = (self.dOmega-self.B)/2.0
        self.BW = (self.A+self.B)/np.pi


class TimeFrequencyTrack:
    def __init__(self, wdm: WDMSetup, time: np.ndarray = None, freq: np.ndarray = None):
        # alias some pieces of the wdm structure
        HBW = wdm.BW/2.0  #half bandwidth of wavelet filter
        Tobs = wdm.NT*wdm.NF*wdm.cadence
        self.wdm = wdm
        n_separate = time.shape[0]

        # spline for t(f)
        tf_spline = CubicSplineInterpolant(freq[None, :, :], time[None, None, :, :])
            # struct CubicSpline *tf_spline = alloc_cubic_spline(N)
            # initialize_cubic_spline(tf_spline, freq, time, SPLINE_BINARY_SEARCH)

        # which frequency layers
        self.min_layer = np.floor((freq[:, 0] - HBW)/WAVELET_BANDWIDTH).astype(int)
        self.max_layer = np.floor(freq[:, -1]/WAVELET_BANDWIDTH).astype(int)
            
        self.min_layer[self.min_layer < 1] = 1
        self.max_layer[self.max_layer > wdm.NF-1] = wdm.NF-1
        
        # mappings for vectorization
        repeats = self.max_layer - self.min_layer
        identification = np.arange(np.prod(self.max_layer.shape[:-1]), dtype=int)
        _min_layer_repeat = np.repeat(self.min_layer.flatten(), repeats.flatten(), axis=-1)
        ident_repeat = np.repeat(identification.flatten(), repeats.flatten(), axis=-1)
        _, uni_index, uni_inverse = np.unique(ident_repeat, return_index=True, return_inverse=True)
        _count = np.arange(_min_layer_repeat.shape[0])
        layers = _count - _count[uni_index][uni_inverse] + _min_layer_repeat

        # for layer in range(self.min_layer, self.max_layer):
        # bandwidth of frequency layer
        fmin = layers*WAVELET_BANDWIDTH - HBW
        fmax = layers*WAVELET_BANDWIDTH + HBW
        
        # duration that signal spends in layer
        tmin = np.zeros_like(fmin)
        tmax = np.zeros_like(fmax)
        # if(fmin>freq[0] and fmin<freq[-1]):
        #     tmin = tf_spline(fmin[None, None, :])  # spline_interpolation(tf_spline, fmin)
        
        # THIS LOOKS LIKE IT COULD LEAVE ISSUES FOR TMAX @TYSON
        run_through_spline_max = np.arange(len(fmax))[(fmax>freq[ident_repeat, 0]) & (fmax<freq[ident_repeat, -1])]
        tmax[run_through_spline_max] = tf_spline.interp_special(fmax[run_through_spline_max], ident_repeat[run_through_spline_max])[0]
        
        run_through_spline_min = np.arange(len(fmin))[(fmin>freq[ident_repeat, 0]) & (fmin<freq[ident_repeat, -1])]
        # [0] at the end because we only have one splined parameter here
        tmin[run_through_spline_min] = tf_spline.interp_special(fmin[run_through_spline_min], ident_repeat[run_through_spline_min])[0]
        
        tmin[(tmin<0.0)] = 0.0
        tmax[(tmax>Tobs)] = Tobs

        # find number of time pixels in the duration, plus some padding, cast to the nearest 2^n
        # TODO: is the 2^n just because you wanted to use a radix2 FFT?
        n  = (np.ceil(tmax/WAVELET_DURATION) - np.floor(tmin/WAVELET_DURATION)).astype(int) + 2.0*wdm.oversample - 1
        n2 = np.pow(2,np.floor(np.log2(n))).astype(int)
                
        n2[(n2 < (n-2))] = n2[(n2 < (n-2))] * 2 #willing to miss the two end pixels in time #TODO: huh?
        
        # find middle pixel relative to start of segment
        i_mid = 0.5*(tmin+tmax)/WAVELET_DURATION #pixel in the middle of the band
        i_mid[(i_mid%2 != 0)] = i_mid[(i_mid%2 != 0)] - 1
        #needs to be even so as to not mess up the transform #TODO: huh?
        
        i_mid[(i_mid-n2/2 < 0)] = n2[(i_mid-n2/2 < 0)] / 2.
        # if(i_mid-n2/2 < 0):
            # i_mid = n2/2 #TODO: huh??

        self.segment_midpt = np.zeros((n_separate, wdm.NF), dtype=np.int32)
        self.segment_size = np.zeros((n_separate, wdm.NF), dtype=np.int32)

        self.segment_size = n2
        self.segment_midpt = i_mid
        self.layers = layers
        self.ident_repeat = ident_repeat


if __name__ == "__main__":

    Tobs = YRSID_SI
    wdm = WDMSetup(Tobs)

    # set parameters
    f_ref = 0.0  # let phenom codes set f_ref -> fmax = max(f^2A(f))
    phi_ref = 0.0  # phase at f_ref
    m1 = 1e6
    m2 = 5e5
    a1 = 0.2
    a2 = 0.4
    dist = 18e3 * PC_SI * 1e6  # 3e3 in Mpc
    inc = np.pi / 3.0
    beta = np.pi / 4.0  # ecliptic latitude
    lam = np.pi / 5.0  # ecliptic longitude
    psi = np.pi / 6.0  # polarization angle
    t_ref = 0.8 * YRSID_SI  # t_ref  (in the SSB reference frame)

    # frequencies to interpolate to
    freq_new = np.logspace(-4, 0, 10000)
    modes = [(2, 2), (2, 1), (3, 3), (3, 2), (4, 4), (4, 3)]
    orbits = EqualArmlengthOrbits()
    orbits.configure(linear_interp_setup=True)

    wave_gen = BBHWaveformFD(
        amp_phase_kwargs=dict(run_phenomd=False),
        response_kwargs=dict(orbits=orbits),
    )

    wave = wave_gen(
        m1,
        m2,
        a1,
        a2,
        dist,
        phi_ref,
        f_ref,
        inc,
        lam,
        beta,
        psi,
        t_ref,
        freqs=freq_new,
        modes=modes,
        direct=False,
        output_splines=True,
        tdi_in_amp_phase=True,
        length=1024,
    )# [0]

    tf_track = TimeFrequencyTrack(wdm, wave.y_shaped[2, 0, 0][None, :], wave.x_shaped[0, 0][None, :])
    wave_out = np.zeros((1, 3, wdm.NF, wdm.NT), dtype=complex)

    from bbhx.utils.transform import LISA_to_SSB
    tSSB, lamSSB, betaSSB, psiSSB = LISA_to_SSB(t_ref, lam, beta, psi)
    
    tc = np.array([tSSB])
    Nspline = wave.y_shaped.shape[-1]
    # fix this is just for 22
    freq_grid = wave.x_shaped[:, 0].reshape(-1, wave.length)
    Nsegment = tf_track.segment_size
    nmid = tf_track.segment_midpt
    delta_f = 1./(Nsegment*WAVELET_DURATION)
    delta_t = Tobs - tc[tf_track.ident_repeat] + (nmid - Nsegment/2)*WAVELET_DURATION
    wave_out[:, :, :, 0] = 0.0
    wave_out[:, :, :, 1] = 0.0

    Nsegment_all = np.repeat(Nsegment, repeats=Nsegment)
    delta_f_all = np.repeat(delta_f, repeats=Nsegment)
    layers_all = np.repeat(tf_track.layers, repeats=Nsegment)
    source_all = np.repeat(tf_track.ident_repeat, repeats=Nsegment)
    delta_t_all = np.repeat(delta_t, repeats=Nsegment)

    identification = np.arange(len(Nsegment))
    ident_repeat = np.repeat(identification.flatten(), Nsegment, axis=-1)
    _, uni_index, uni_inverse = np.unique(ident_repeat, return_index=True, return_inverse=True)
    _count = np.arange(Nsegment_all.shape[0])
    i_arr = _count - _count[uni_index][uni_inverse] + 1

    f = (i_arr - Nsegment_all / 2.) * delta_f_all + layers_all * WAVELET_BANDWIDTH

    keep = (f > freq_grid[source_all, 0]) & (f < freq_grid[source_all, -1])
    _out_vals = wave.interp_special(f[keep], source_all[keep]) # *AmpSSB
    Amp = _out_vals[0]
    Phase = _out_vals[1]
    Phase  = 2 * np.pi * f[keep] * delta_t_all[keep] - Phase

    # NEED TO ADJUST TO CHANNELS
    wave_out[(source_all[keep], np.zeros_like(source_all[keep]), layers_all[keep], i_arr[keep])] = Amp * np.exp(1j * Phase)  # MINUS SIGN HERE?
    
    layers_transformed = np.fft.fftshift(np.fft.fft(wave_out, axis=-1))
    # for(i=1 i<Nsegment i++)
    # {
    #     waveform[2*i]   = 0.0
    #     waveform[2*i+1] = 0.0

    #     f = (double)(i - Nsegment/2)*delta_f + layer*WAVELET_BANDWIDTH

    #     if(f>freq_grid[0] && f<freq_grid[Nspline-1])
    #     {
    # AmpSSB = spline_interpolation(amp_ssb_spline,f)
    
    # Amp    = spline_interpolation(amp_tdi_spline,f)*AmpSSB
    # Phase  = spline_interpolation(phase_tdi_spline,f)
    # Phase  = PI2 * f * delta_t - Phase
    
    # waveform[2*i]   = Amp * cos(Phase)
    # waveform[2*i+1] = Amp * sin(Phase)

    breakpoint()
    # plt.loglog(freq_new, np.abs(wave[0]))
    # plt.savefig("check0.png")
    breakpoint()
