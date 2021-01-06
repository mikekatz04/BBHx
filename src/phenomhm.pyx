import numpy as np
cimport numpy as np

from phenomhm.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "PhenomHM.hh":

    void waveform_amp_phase(
        double* waveformOut,
        int* ells_in,
        int* mms_in,
        double* freqs,
        double* m1_SI,
        double* m2_SI,
        double* chi1z,
        double* chi2z,
        double* distance,
        double* phiRef,
        double* f_ref,
        int numModes,
        int length,
        int numBinAll
    );

@pointer_adjust
def waveform_amp_phase_wrap(
    waveformOut,
    ells,
    mms,
    freqs,
    m1_SI,
    m2_SI,
    chi1z,
    chi2z,
    distance,
    phiRef,
    f_ref,
    numModes,
    length,
    numBinAll
):

    cdef size_t waveformOut_in = waveformOut
    cdef size_t ells_in = ells
    cdef size_t mms_in = mms
    cdef size_t freqs_in = freqs
    cdef size_t m1_SI_in = m1_SI
    cdef size_t m2_SI_in = m2_SI
    cdef size_t chi1z_in = chi1z
    cdef size_t chi2z_in = chi2z
    cdef size_t distance_in = distance
    cdef size_t phiRef_in = phiRef
    cdef size_t f_ref_in = f_ref

    waveform_amp_phase(
        <double*> waveformOut_in,
        <int*> ells_in,
        <int*> mms_in,
        <double*> freqs_in,
        <double*> m1_SI_in,
        <double*> m2_SI_in,
        <double*> chi1z_in,
        <double*> chi2z_in,
        <double*> distance_in,
        <double*> phiRef_in,
        <double*> f_ref_in,
        numModes,
        length,
        numBinAll
    )

    return
