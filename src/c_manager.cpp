/*
This is the central piece of code. This file implements a class
that takes data in on the cpu side, copies
it to the gpu, and exposes functions that let
you perform actions with the GPU

This class will get translated into python via cython
*/

#include <c_manager.h>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>

using namespace std;

PhenomHM::PhenomHM (int max_length_init_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_,
    double *data_freqs_,
    cmplx *data_channel1_,
    cmplx *data_channel2_,
    cmplx *data_channel3_, int data_stream_length_, double *channel1_ASDinv_, double *channel2_ASDinv_, double *channel3_ASDinv_, int TDItag_){

    max_length_init = max_length_init_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
    data_freqs = data_freqs_;
    data_stream_length = data_stream_length_;
    channel1_ASDinv = channel1_ASDinv_;
    channel2_ASDinv = channel2_ASDinv_;
    channel3_ASDinv = channel3_ASDinv_;
    data_channel1 = data_channel1_;
    data_channel2 = data_channel2_;
    data_channel3 = data_channel3_;

    TDItag = TDItag_;
    to_gpu = 0;

    // DECLARE ALL THE  NECESSARY STRUCTS
    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

    mode_vals = cpu_create_modes(num_modes, l_vals, m_vals, max_length_init, to_gpu, 1);

    cShift = new double[7];

    cShift[0] = 0.0;
    cShift[1] = PI_2; /* i shift */
    cShift[2] = 0.0;
    cShift[3] = -PI_2;/* -i shift */
    cShift[4] = PI;/* 1 shift */
    cShift[5] = PI_2;/* -1 shift */
    cShift[6] = 0.0;

  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;
}

void PhenomHM::gen_amp_phase(double *freqs_, int current_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double phiRef_,
    double fRef_){

    assert(current_length_ <= max_length_init);

    // for phenomHM internal calls
    deltaF = -1.0;

    for (int i=0; i<num_modes; i++){
        mode_vals[i].length = current_length;
    }
    freqs = freqs_;
    current_length = current_length_;
    m1 = m1_;
    m2 = m2_;
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    phiRef = phiRef_;
    fRef = fRef_;

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs,
        current_length,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        phiRef,
        deltaF,
        fRef,
        num_modes,
        to_gpu,
        pHM_trans,
        pAmp_trans,
        amp_prefactors_trans,
        pDPreComp_all_trans,
        q_all_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in
}

/*void PhenomHM::LISAresponseFD(){
    int order_fresnel_stencil = 0;
    JustLISAFDresponseTDI_wrap(mode_vals, H_mat, frqs, old_freqs, d_log10f, l_vals, m_vals, num_modes, num_points, inc, lam, beta, psi, phi0, t0, tRef, merger_freq, TDItag, order_fresnel_stencil)
}*/

void PhenomHM::GetAmpPhase(double* amp_, double* phase_){
    assert(current_status >= 1);

    for (int mode_i=0; mode_i<num_modes; mode_i++){
        /*for (int i=0; i<current_length; i++){
            printf("%e, %e\n", mode_vals[mode_i].amp[i], mode_vals[mode_i].phase[i]);
        }*/
        memcpy(&amp_[mode_i*current_length], mode_vals[mode_i].amp, current_length*sizeof(double));
        memcpy(&phase_[mode_i*current_length], mode_vals[mode_i].phase, current_length*sizeof(double));
    }
}

PhenomHM::~PhenomHM() {
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete[] pDPreComp_all_trans;
  delete[] q_all_trans;
  delete[] cShift;
  cpu_destroy_modes(mode_vals);
}
