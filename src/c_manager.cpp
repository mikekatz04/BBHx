
#include <c_manager.h>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>

using namespace std;


GPUPhenomHM::GPUPhenomHM (int max_length_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_){

    max_length = max_length_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;

    // DECLARE ALL THE  NECESSARY STRUCTS
    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

    to_gpu = 0;
    to_interp = 0;

  mode_vals = cpu_create_modes(num_modes, l_vals, m_vals, max_length, to_gpu, to_interp);

  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;
}


void GPUPhenomHM::cpu_gen_PhenomHM(double *freqs_, int f_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    freqs = freqs_;
    f_length = f_length_;
    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    inclination = inclination_;
    phiRef = phiRef_;
    deltaF = deltaF_;
    f_ref = f_ref_;

    for (int i=0; i<num_modes; i++){
        mode_vals[i].length = f_length;
    }

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs,
        f_length,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        inclination,
        phiRef,
        deltaF,
        f_ref,
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


void GPUPhenomHM::Get_Waveform (int mode_i, double* amp_, double* phase_) {
    assert(to_gpu == 0);
    memcpy(amp_, mode_vals[mode_i].amp, f_length*sizeof(double));
    memcpy(phase_, mode_vals[mode_i].phase, f_length*sizeof(double));
}

GPUPhenomHM::~GPUPhenomHM() {
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete pDPreComp_all_trans;
  delete q_all_trans;
  cpu_destroy_modes(mode_vals);
}
