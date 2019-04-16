/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <c_manager.h>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include "PhenomHM.h"
#include <complex>
#include <numeric> 


using namespace std;

PhenomHM::PhenomHM (double *freqs_,
    int f_length_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_){

    freqs = freqs_;
    f_length = f_length_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;

    f_length = f_length_;

    // DECLARE ALL THE  NECESSARY STRUCTS

    freqs_geom_trans = new double[f_length];

    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

    factorp_trans = new std::complex<double>[num_modes];
    factorc_trans = new std::complex<double>[num_modes];


    hptilde = new std::complex<double>[num_modes*f_length];
    hctilde = new std::complex<double>[num_modes*f_length];




  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;
}




void PhenomHM::gen_PhenomHM(
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    inclination = inclination_;
    phiRef = phiRef_;
    deltaF = deltaF_;
    f_ref = f_ref_;

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    int to_gpu = 0;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        hptilde,
        hctilde,
        freqs,
        freqs_geom_trans,
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
        l_vals,
        m_vals,
        num_modes,
        to_gpu,
        pHM_trans,
        pAmp_trans,
        amp_prefactors_trans,
        pDPreComp_all_trans,
        q_all_trans,
        factorp_trans,
        factorc_trans,
        &t0,
        &phi0,
        &amp0);
    assert (retcode == 1); //,PD_EFUNC, "IMRPhenomHMCore failed in
}

double PhenomHM::Likelihood (){

    // need to fix this
    std::complex<double> init = std::complex<double>(0.0, 0.0);
    std::complex<double> * hptilde_conj = new std::complex<double>[f_length*num_modes];
    int i;
    //conjugate
    for (i=0; i<f_length*num_modes; i++) hptilde_conj[i] = std::complex<double>(hptilde[i].real(), -1.0*hptilde[i].imag());
    std::complex<double> result = std::inner_product(hptilde_conj, hptilde_conj + (f_length*num_modes), hptilde, init);
    delete hptilde_conj;
    return real(result);
}

void PhenomHM::Get_Waveform (std::complex<double>* hptilde_, std::complex<double>* hctilde_) {
 memcpy(hptilde_, hptilde, num_modes*f_length*sizeof(std::complex<double>));
 memcpy(hctilde_, hctilde, num_modes*f_length*sizeof(std::complex<double>));

}

PhenomHM::~PhenomHM() {
  delete freqs_geom_trans;
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete pDPreComp_all_trans;
  delete q_all_trans;
  delete factorp_trans;
  delete factorc_trans;

  delete hptilde;
  delete hctilde;
}
