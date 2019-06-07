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
#include "fdresponse.h"
#include "c_interpolate.h"

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

    H_mat = new cmplx[num_modes*9];

    template_channel1 = new cmplx[num_modes*data_stream_length];
    template_channel2 = new cmplx[num_modes*data_stream_length];
    template_channel3 = new cmplx[num_modes*data_stream_length];

    B = new double[8*data_stream_length*num_modes];

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

void PhenomHM::setup_interp_wave(){
    host_fill_B_wave(mode_vals, B, current_length, num_modes);
    interp.prep(B, current_length, 2*num_modes, 0);
    host_set_spline_constants_wave(mode_vals, B, current_length, num_modes);
}

void PhenomHM::LISAresponseFD(double inc_, double lam_, double beta_, double psi_, double t0_epoch_, double tRef_, double merger_freq_){
    inc = inc_;
    lam = lam_;
    beta = beta_;
    psi = psi_;
    t0_epoch = t0_epoch_;
    tRef = tRef_;
    merger_freq = merger_freq_;
    int order_fresnel_stencil = 0;
    prep_H_info(H_mat, l_vals, m_vals, num_modes, inc, lam, beta, psi, phiRef);
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);
    JustLISAFDresponseTDI_wrap(mode_vals, H_mat, freqs, freqs, d_log10f, l_vals, m_vals, num_modes, current_length, inc, lam, beta, psi, phiRef, t0_epoch, tRef, merger_freq, TDItag, order_fresnel_stencil);
}

void PhenomHM::setup_interp_response(){
    host_fill_B_response(mode_vals, B, current_length, num_modes);
    interp.prep(B, current_length, 8*num_modes, 0);
    host_set_spline_constants_response(mode_vals, B, current_length, num_modes);
}


void PhenomHM::perform_interp(){
    double d_log10f = log10(freqs[1]) - log10(freqs[0]);
    host_interpolate(template_channel1, template_channel2, template_channel3, mode_vals, num_modes, d_log10f, freqs, current_length, data_freqs, data_stream_length, t0_epoch, tRef, channel1_ASDinv, channel2_ASDinv, channel3_ASDinv);
}



void host_combine(cmplx *channel1_out, cmplx *channel2_out, cmplx *channel3_out, ModeContainer* old_mode_vals, int num_modes, double d_log10f, double *old_freqs, int old_length, double *data_freqs, int length, double t0, double tRef, double *X_ASD_inv, double *Y_ASD_inv, double *Z_ASD_inv){

    double f, t, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3;
    double amp, phase, phaseRdelay, phasetimeshift;
    double transferL1_re, transferL1_im, transferL2_re, transferL2_im, transferL3_re, transferL3_im;
    cmplx fastPart;
    cmplx I(0.0, 1.0);
    double f_min_limit = old_freqs[0];
    double f_max_limit = old_freqs[old_length-1];
    int old_ind_below;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<length; i++){
            t = old_mode_vals[mode_i].time_freq_corr[i];
            if (t < 0.0){
                channel1_out[mode_i*length + i] = 0.0+I*0.0;
                channel2_out[mode_i*length + i] = 0.0+I*0.0;
                channel3_out[mode_i*length + i] = 0.0+I*0.0;
                continue;
            }

            f = data_freqs[i];
            /*f = f_min + df * i;
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            if ((old_ind_below == old_length -1) || (f >= f_max_limit) || (f < f_min_limit)){
                channel1_out[mode_i*length + i] = 0.0+0.0*I;
                channel2_out[mode_i*length + i] = 0.0+0.0*I;
                channel3_out[mode_i*length + i] = 0.0+0.0*I;
                continue;
            }
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x*x2;
            // interp amplitude
            coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
            if (coeff_0 < 1e-50){
                channel1_out[mode_i*length + i] = cmplx(0.0, 0.0);
                channel2_out[mode_i*length + i] = cmplx(0.0, 0.0);
                channel3_out[mode_i*length + i] = cmplx(0.0, 0.0);
                continue;
            }
            coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

            amp = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);*/
            amp = old_mode_vals[mode_i].amp[i];
            if (amp < 1e-40){
                channel1_out[mode_i*length + i] = 0.0+I*0.0;
                channel2_out[mode_i*length + i] = 0.0+I*0.0;
                channel3_out[mode_i*length + i] = 0.0+I*0.0;
                continue;
            }

            // interp phase
            /*coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

            phase  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].phaseRdelay[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].phaseRdelay_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].phaseRdelay_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].phaseRdelay_coeff_3[old_ind_below];

            phaseRdelay  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);*/

            phase = old_mode_vals[mode_i].phase[i];
            phaseRdelay = old_mode_vals[mode_i].phaseRdelay[i];
            phasetimeshift = 2.*PI*(t0+tRef)*f;
            fastPart = amp * exp(I*(phase + phaseRdelay + phasetimeshift));


            // X
            /*coeff_0 = old_mode_vals[mode_i].transferL1_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_re_coeff_3[old_ind_below];

            transferL1_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL1_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL1_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL1_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL1_im_coeff_3[old_ind_below];

            transferL1_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);*/

            transferL1_re = old_mode_vals[mode_i].transferL1_re[i];
            transferL1_im = old_mode_vals[mode_i].transferL1_im[i];

            channel1_out[mode_i*length + i] = ((transferL1_re+I*transferL1_im) * fastPart * X_ASD_inv[i]);

            // Y
            /*coeff_0 = old_mode_vals[mode_i].transferL2_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_re_coeff_3[old_ind_below];

            transferL2_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL2_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL2_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL2_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL2_im_coeff_3[old_ind_below];

            transferL2_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);*/

            transferL2_re = old_mode_vals[mode_i].transferL2_re[i];
            transferL2_im = old_mode_vals[mode_i].transferL2_im[i];

            channel2_out[mode_i*length + i] = ((transferL2_re+I*transferL2_im) * fastPart * Y_ASD_inv[i]);

            // Z
            /*coeff_0 = old_mode_vals[mode_i].transferL3_re[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_re_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_re_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_re_coeff_3[old_ind_below];

            transferL3_re  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

            coeff_0 = old_mode_vals[mode_i].transferL3_im[old_ind_below];
            coeff_1 = old_mode_vals[mode_i].transferL3_im_coeff_1[old_ind_below];
            coeff_2 = old_mode_vals[mode_i].transferL3_im_coeff_2[old_ind_below];
            coeff_3 = old_mode_vals[mode_i].transferL3_im_coeff_3[old_ind_below];

            transferL3_im  = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);*/
            transferL3_re = old_mode_vals[mode_i].transferL3_re[i];
            transferL3_im = old_mode_vals[mode_i].transferL3_im[i];

            channel3_out[mode_i*length + i] = ((transferL3_re+I*transferL3_im) * fastPart * Z_ASD_inv[i]);
        }
    }
}


void PhenomHM::Combine(){
    double d_log10f = log10(data_freqs[1]) - log10(data_freqs[0]);
    host_combine(template_channel1, template_channel2, template_channel3, mode_vals, num_modes, d_log10f, freqs, current_length, data_freqs, data_stream_length, t0_epoch, tRef, channel1_ASDinv, channel2_ASDinv, channel3_ASDinv);
}

cmplx complex_dot_d_h(cmplx *arr1, cmplx *arr2, int n, int num_modes){
    // arr1 template arr2 data
    cmplx I(0.0, 1.0);
    cmplx sum = 0.0 + 0.0*I;
    cmplx temp_sum = 0.0 + 0.0*I;
    for (int i=0; i<n; i++){
        temp_sum = 0.0 + 0.0*I;
        for (int mode_i=0; mode_i<num_modes; mode_i++){
            temp_sum += arr1[mode_i*n + i];
        }
        sum += std::conj(arr2[i])*temp_sum;
    }
    return sum;
}

cmplx complex_dot_h_h(cmplx *arr1, cmplx *arr2, int n, int num_modes){
    // arr1 template arr2 template
    cmplx I(0.0, 1.0);
    cmplx sum = 0.0 + 0.0*I;
    cmplx temp_sum = 0.0 + 0.0*I;
    for (int i=0; i<n; i++){
        temp_sum = 0.0 + 0.0*I;
        for (int mode_i=0; mode_i<num_modes; mode_i++){
            temp_sum += arr1[mode_i*n + i];
        }
        sum += std::conj(temp_sum)*temp_sum;
    }
    return sum;
}

void PhenomHM::Likelihood (double *like_out_){

     assert(current_status == 5);
     double d_h = 0.0;
     double h_h = 0.0;
     cmplx res, result;

     result = complex_dot_d_h(template_channel1, data_channel1, data_stream_length, num_modes);
     d_h += result.real();

     result = complex_dot_d_h(template_channel2, data_channel2, data_stream_length, num_modes);
     d_h += result.real();

     result = complex_dot_d_h(template_channel3, data_channel3, data_stream_length, num_modes);
     d_h += result.real();

     result = complex_dot_h_h(template_channel1, template_channel1, data_stream_length, num_modes);
     h_h += result.real();

     result = complex_dot_h_h(template_channel2, template_channel2, data_stream_length, num_modes);
     h_h += result.real();

     result = complex_dot_h_h(template_channel3, template_channel3, data_stream_length, num_modes);
     h_h += result.real();

     like_out_[0] = 4*d_h;
     like_out_[1] = 4*h_h;
}

void PhenomHM::GetTDI(cmplx *template_channel1_, cmplx *template_channel2_, cmplx *template_channel3_){
    memcpy(template_channel1_, template_channel1, num_modes*data_stream_length*sizeof(cmplx));
    memcpy(template_channel2_, template_channel2, num_modes*data_stream_length*sizeof(cmplx));
    memcpy(template_channel3_, template_channel3, num_modes*data_stream_length*sizeof(cmplx));
}

void PhenomHM::GetAmpPhase(double* amp_, double* phase_){
    assert(current_status >= 1);

    for (int mode_i=0; mode_i<num_modes; mode_i++){
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
  delete[] H_mat;
  delete[] template_channel1;
  delete[] template_channel2;
  delete[] template_channel3;
  delete[] B;
}
