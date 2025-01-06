import numpy as np
cimport numpy as np

from bbhx.utils.utility import pointer_adjust

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
        double* f_ref,
        int numModes,
        int length,
        int numBinAll,
        double* Mf_RD_lm_all,
        double* Mf_DM_lm_all,
        int run_phenomd
    )

    void get_phenomhm_ringdown_frequencies_wrap(
        double *fringdown,
        double *fdamp,
        double *m1,
        double *m2,
        double *chi1z,
        double *chi2z,
        int *ells_in,
        int *mm_in,
        int numModes,
        int numBinAll
    )

    void get_phenomd_ringdown_frequencies_wrap(
        double *fringdown,
        double *fdamp,
        double *m1,
        double *m2,
        double *chi1z,
        double *chi2z,
        int numBinAll,
        double *y_rd_all,
        double *c1_rd_all,
        double *c2_rd_all,
        double *c3_rd_all,
        double *y_dm_all,
        double *c1_dm_all,
        double *c2_dm_all,
        double *c3_dm_all,
        double dspin
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
    f_ref,
    numModes,
    length,
    numBinAll,
    Mf_RD_lm_all,
    Mf_DM_lm_all,
    run_phenomd
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
    cdef size_t f_ref_in = f_ref
    cdef size_t Mf_RD_lm_all_in = Mf_RD_lm_all
    cdef size_t Mf_DM_lm_all_in = Mf_DM_lm_all


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
        <double*> f_ref_in,
        numModes,
        length,
        numBinAll,
        <double*> Mf_RD_lm_all_in,
        <double*> Mf_DM_lm_all_in,
        run_phenomd
    )

    return

@pointer_adjust
def get_phenomhm_ringdown_frequencies(
    fringdown,
    fdamp,
    m1,
    m2,
    chi1z,
    chi2z,
    ells,
    mms,
    numModes,
    numBinAll,
):

    cdef size_t fringdown_in = fringdown
    cdef size_t fdamp_in = fdamp
    cdef size_t m1_in = m1
    cdef size_t m2_in = m2
    cdef size_t chi1z_in = chi1z
    cdef size_t chi2z_in = chi2z
    cdef size_t ells_in = ells
    cdef size_t mms_in = mms

    get_phenomhm_ringdown_frequencies_wrap(
        <double *>fringdown_in,
        <double *>fdamp_in,
        <double *>m1_in,
        <double *>m2_in,
        <double *>chi1z_in,
        <double *> chi2z_in,
        <int *> ells_in,
        <int *> mms_in,
        numModes,
        numBinAll
    )
    return

@pointer_adjust
def get_phenomd_ringdown_frequencies(
    fringdown,
    fdamp,
    m1,
    m2,
    chi1z,
    chi2z,
    numBinAll,
    y_rd_all,
    c1_rd_all,
    c2_rd_all,
    c3_rd_all,
    y_dm_all,
    c1_dm_all,
    c2_dm_all,
    c3_dm_all,
    dspin
):

    cdef size_t fringdown_in = fringdown
    cdef size_t fdamp_in = fdamp
    cdef size_t m1_in = m1
    cdef size_t m2_in = m2
    cdef size_t chi1z_in = chi1z
    cdef size_t chi2z_in = chi2z
    cdef size_t y_rd_all_in = y_rd_all
    cdef size_t c1_rd_all_in = c1_rd_all
    cdef size_t c2_rd_all_in = c2_rd_all
    cdef size_t c3_rd_all_in = c3_rd_all
    cdef size_t y_dm_all_in = y_dm_all
    cdef size_t c1_dm_all_in = c1_dm_all
    cdef size_t c2_dm_all_in = c2_dm_all
    cdef size_t c3_dm_all_in = c3_dm_all



    get_phenomd_ringdown_frequencies_wrap(
        <double*> fringdown_in,
        <double*> fdamp_in,
        <double*> m1_in,
        <double*> m2_in,
        <double*> chi1z_in,
        <double*> chi2z_in,
        numBinAll,
        <double*> y_rd_all_in,
        <double*> c1_rd_all_in,
        <double*> c2_rd_all_in,
        <double*> c3_rd_all_in,
        <double*> y_dm_all_in,
        <double*> c1_dm_all_in,
        <double*> c2_dm_all_in,
        <double*> c3_dm_all_in,
        dspin
    )
    return
