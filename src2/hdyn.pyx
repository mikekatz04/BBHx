import numpy as np
cimport numpy as np

from phenomhm.utils.utility import pointer_adjust

assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "full.h":
    ctypedef void* cmplx 'cmplx'

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

    void LISA_response(
        double* response_out,
        int* ells_in,
        int* mms_in,
        double* freqs,
        double* phiRef,
        double* f_ref,
        double* inc,
        double* lam,
        double* beta,
        double* psi,
        double* tRef_wave_frame,
        double* tRef_sampling_frame,
        double tBase, int TDItag, int order_fresnel_stencil,
        int numModes,
        int length,
        int numBinAll,
        int includesAmps
    );

    void interpolate(double* freqs, double* propArrays,
                     double* B, double* upper_diag, double* diag, double* lower_diag,
                     int length, int numInterpParams, int numModes, int numBinAll);

    void InterpTDI(long* templateChannels_ptrs, double* dataFreqs, double dlog10f, double* freqs, double* propArrays, double* c1, double* c2, double* c3, double* t_mrg, double* t_start, double* t_end, int length, int data_length, int numBinAll, int numModes, double t_obs_start, double t_obs_end, long* inds_ptrs, int* inds_start, int* ind_lengths);

    void hdyn(cmplx* likeOut1, cmplx* likeOut2,
                        cmplx* templateChannels, cmplx* dataConstants,
                        double* dataFreqs,
                        int numBinAll, int data_length, int nChannels);

    void direct_sum(cmplx* channel1, cmplx* channel2, cmplx* channel3,
                    double* bbh_buffer,
                    int numBinAll, int data_length, int nChannels, int numModes);


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

@pointer_adjust
def LISA_response_wrap(
     response_out,
     ells,
     mms,
     freqs,
     phiRef,
     f_ref,
     inc,
     lam,
     beta,
     psi,
     tRef_wave_frame,
     tRef_sampling_frame,
    tBase, TDItag, order_fresnel_stencil,
    numModes,
    length,
    numBinAll,
    includesAmps
):

    cdef size_t response_out_in = response_out
    cdef size_t ells_in = ells
    cdef size_t mms_in = mms
    cdef size_t freqs_in = freqs
    cdef size_t inc_in = inc
    cdef size_t lam_in = lam
    cdef size_t beta_in = beta
    cdef size_t psi_in = psi
    cdef size_t tRef_wave_frame_in = tRef_wave_frame
    cdef size_t tRef_sampling_frame_in = tRef_sampling_frame
    cdef size_t phiRef_in = phiRef
    cdef size_t f_ref_in = f_ref

    LISA_response(
        <double*> response_out_in,
        <int*> ells_in,
        <int*> mms_in,
        <double*> freqs_in,
        <double*> phiRef_in,
        <double*> f_ref_in,
        <double*> inc_in,
        <double*> lam_in,
        <double*> beta_in,
        <double*> psi_in,
        <double*> tRef_wave_frame_in,
        <double*> tRef_sampling_frame_in,
        tBase, TDItag, order_fresnel_stencil,
        numModes,
        length,
        numBinAll,
        includesAmps
    )

@pointer_adjust
def interpolate_wrap(freqs, propArrays,
                     B, upper_diag, diag, lower_diag,
                     length, numInterpParams, numModes, numBinAll):

    cdef size_t freqs_in = freqs
    cdef size_t propArrays_in = propArrays
    cdef size_t B_in = B
    cdef size_t upper_diag_in = upper_diag
    cdef size_t diag_in = diag
    cdef size_t lower_diag_in = lower_diag

    interpolate(<double*>freqs_in, <double*>propArrays_in,
              <double*>B_in, <double*>upper_diag_in, <double*>diag_in, <double*>lower_diag_in,
              length, numInterpParams, numModes, numBinAll)

@pointer_adjust
def InterpTDI_wrap(templateChannels_ptrs, dataFreqs, dlog10f, freqs, propArrays, c1, c2, c3, t_mrg, t_start, t_end, length, data_length, numBinAll, numModes, t_obs_start, t_obs_end, inds_ptrs, inds_start, ind_lengths):

    cdef size_t freqs_in = freqs
    cdef size_t propArrays_in = propArrays
    cdef size_t templateChannels_ptrs_in = templateChannels_ptrs
    cdef size_t dataFreqs_in = dataFreqs
    cdef size_t c1_in = c1
    cdef size_t c2_in = c2
    cdef size_t c3_in = c3
    cdef size_t t_mrg_in = t_mrg
    cdef size_t t_start_in = t_start
    cdef size_t t_end_in = t_end
    cdef size_t inds_ptrs_in = inds_ptrs
    cdef size_t inds_start_in = inds_start
    cdef size_t ind_lengths_in = ind_lengths

    InterpTDI(<long*> templateChannels_ptrs_in, <double*> dataFreqs_in, dlog10f, <double*> freqs_in, <double*> propArrays_in, <double*> c1_in, <double*> c2_in, <double*> c3_in, <double*> t_mrg_in, <double*> t_start_in, <double*> t_end_in, length, data_length, numBinAll, numModes, t_obs_start, t_obs_end, <long*> inds_ptrs_in, <int*> inds_start_in, <int*> ind_lengths_in);


@pointer_adjust
def hdyn_wrap(likeOut1, likeOut2,
                    templateChannels, dataConstants,
                    dataFreqs,
                    numBinAll, data_length, nChannels):

    cdef size_t likeOut1_in = likeOut1
    cdef size_t likeOut2_in = likeOut2
    cdef size_t templateChannels_in = templateChannels
    cdef size_t dataConstants_in = dataConstants
    cdef size_t dataFreqs_in = dataFreqs

    hdyn(<cmplx*> likeOut1_in, <cmplx*> likeOut2_in,
            <cmplx*> templateChannels_in, <cmplx*> dataConstants_in,
            <double*> dataFreqs_in,
            numBinAll, data_length, nChannels);

@pointer_adjust
def direct_sum_wrap(channel1, channel2, channel3,
                bbh_buffer,
                numBinAll, data_length, nChannels, numModes):

    cdef size_t channel1_in = channel1
    cdef size_t channel2_in = channel2
    cdef size_t channel3_in = channel3
    cdef size_t bbh_buffer_in = bbh_buffer

    direct_sum(<cmplx*> channel1_in, <cmplx*> channel2_in, <cmplx*> channel3_in,
                    <double*> bbh_buffer_in,
                    numBinAll, data_length, nChannels, numModes)
