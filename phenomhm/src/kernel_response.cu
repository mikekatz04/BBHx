/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This code computes the fast Fourier domain response function for LISA
 *  based on Marsat and Baker 2018. This code contains the GPU functions
 *  for this calculation. See fdresponse.cpp for comments. See pyFDresponse.py in LDC.
 *
 *
 *  Copyright (C) 2019 Michael Katz
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */

#include "globalPhenomHM.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "fdresponse.h"

using namespace std;

typedef struct tagd_Gslr_holder{
    agcmplx G21;
    agcmplx G12;
    agcmplx G23;
    agcmplx G32;
    agcmplx G31;
    agcmplx G13;
} d_Gslr_holder;

typedef struct tagd_transferL_holder{
    agcmplx transferL1;
    agcmplx transferL2;
    agcmplx transferL3;
    double phaseRdelay;
} d_transferL_holder;

CUDA_CALLABLE_MEMBER
double d_sinc(double x){
    if (x == 0.0) return 1.0;
    else return sin(x)/x;
}

CUDA_CALLABLE_MEMBER
double d_dot_product_1d(double arr1[3], double arr2[3]){
    double out = 0.0;
    for (int i=0; i<3; i++){
        out += arr1[i]*arr2[i];
    }
    return out;
}

CUDA_CALLABLE_MEMBER
agcmplx d_vec_H_vec_product(double arr1[3], agcmplx *H, double arr2[3]){
    agcmplx c_arr1[3] = {agcmplx(arr1[0], 0.0),
                                 agcmplx(arr1[1], 0.0),
                                 agcmplx(arr1[2], 0.0)};

    agcmplx c_arr2[3] = {agcmplx(arr2[0], 0.0),
                                 agcmplx(arr2[1], 0.0),
                                 agcmplx(arr2[2], 0.0)};

    agcmplx I(0.0, 1.0);
    agcmplx out(0.0, 0.0);
    agcmplx trans(0.0, 0.0);
    for (int i=0; i<3; i++){
        trans = agcmplx(0.0, 0.0);
        for (int j=0; j<3; j++){
            trans += (H[i*3 + j] * c_arr2[j]);
        }
        out += c_arr1[i]*trans;
    }
    return out;
}

/* # Single-link response
# 'full' does include the orbital-delay term, 'constellation' does not
 */
CUDA_CALLABLE_MEMBER
d_Gslr_holder d_EvaluateGslr(double t, double f, agcmplx *H, double k[3], int response){
    // response == 1 is full ,, response anything else is constellation
    //# Trajectories, p0 used only for the full response
    agcmplx I(0.0, 1.0);
    agcmplx m_I(0.0, -1.0);
    double alpha = Omega0*t; double c = cos(alpha); double s = sin(alpha);
    double a = aorbit; double e = eorbit;

    double p0[3] = {a*c, a*s, 0.*t}; // funcp0(t)
    double p1L[3] = {- a*e*(1 + s*s), a*e*c*s, -a*e*sqrt3*c}; //funcp1L(t)
    double p2L[3] = {a*e/2*(sqrt3*c*s + (1 + s*s)), a*e/2*(-c*s - sqrt3*(1 + c*c)), -a*e*sqrt3/2*(sqrt3*s - c)}; //funcp2L(t)
    double p3L[3] = {a*e/2*(-sqrt3*c*s + (1 + s*s)), a*e/2*(-c*s + sqrt3*(1 + c*c)), -a*e*sqrt3/2*(-sqrt3*s - c)}; //funcp3L(t)
    double n1[3] = {-1./2*c*s, 1./2*(1 + c*c), sqrt3/2*s}; //funcn1(t)
    double n2[3] = {c*s - sqrt3*(1 + s*s), sqrt3*c*s - (1 + c*c), -sqrt3*s - 3*c}; //funcn2(t)
    for (int i=0; i<3; i++) n2[i] = n2[i]*1./4.;
    double n3[3] = {c*s + sqrt3*(1 + s*s), -sqrt3*c*s - (1 + c*c), -sqrt3*s + 3*c}; //funcn3(t)
    for (int i=0; i<3; i++) n3[i] = n3[i]*1./4.;
    // # Compute intermediate scalar products
    // t scalar case
    double kn1 = d_dot_product_1d(k, n1);
    double kn2 = d_dot_product_1d(k, n2);
    double kn3 = d_dot_product_1d(k, n3);

    agcmplx n1Hn1 = d_vec_H_vec_product(n1, H, n1); //np.dot(n1, np.dot(H, n1))
    agcmplx n2Hn2 = d_vec_H_vec_product(n2, H, n2); //np.dot(n2, np.dot(H, n2))
    agcmplx n3Hn3 = d_vec_H_vec_product(n3, H, n3); //np.dot(n3, np.dot(H, n3))

    double p1L_plus_p2L[3] = {p1L[0]+p2L[0], p1L[1]+p2L[1], p1L[2]+p2L[2]};
    double p2L_plus_p3L[3] = {p2L[0]+p3L[0], p2L[1]+p3L[1], p2L[2]+p3L[2]};
    double p3L_plus_p1L[3] = {p3L[0]+p1L[0], p3L[1]+p1L[1], p3L[2]+p1L[2]};

    double kp1Lp2L = d_dot_product_1d(k, p1L_plus_p2L);
    double kp2Lp3L = d_dot_product_1d(k, p2L_plus_p3L);
    double kp3Lp1L = d_dot_product_1d(k, p3L_plus_p1L);
    double kp0 = d_dot_product_1d(k, p0);

    // # Prefactors - projections are either scalars or vectors
    agcmplx factorcexp0;
    if (response==1) factorcexp0 = gcmplx::exp(I*2.*PI*f/C_SI * kp0); // I*2.*PI*f/C_SI * kp0
    else factorcexp0 = agcmplx(1.0, 0.0);
    double prefactor = PI*f*L_SI/C_SI;

    agcmplx factorcexp12 = gcmplx::exp(I*prefactor * (1.+kp1Lp2L/L_SI)); //prefactor * (1.+kp1Lp2L/L_SI)
    agcmplx factorcexp23 = gcmplx::exp(I*prefactor * (1.+kp2Lp3L/L_SI)); //prefactor * (1.+kp2Lp3L/L_SI)
    agcmplx factorcexp31 = gcmplx::exp(I*prefactor * (1.+kp3Lp1L/L_SI)); //prefactor * (1.+kp3Lp1L/L_SI)

    agcmplx factorsinc12 = d_sinc( prefactor * (1.-kn3));
    agcmplx factorsinc21 = d_sinc( prefactor * (1.+kn3));
    agcmplx factorsinc23 = d_sinc( prefactor * (1.-kn1));
    agcmplx factorsinc32 = d_sinc( prefactor * (1.+kn1));
    agcmplx factorsinc31 = d_sinc( prefactor * (1.-kn2));
    agcmplx factorsinc13 = d_sinc( prefactor * (1.+kn2));

    // # Compute the Gslr - either scalars or vectors
    d_Gslr_holder Gslr_out;

    agcmplx commonfac = I*prefactor*factorcexp0;
    Gslr_out.G12 = commonfac * n3Hn3 * factorsinc12 * factorcexp12;
    Gslr_out.G21 = commonfac * n3Hn3 * factorsinc21 * factorcexp12;
    Gslr_out.G23 = commonfac * n1Hn1 * factorsinc23 * factorcexp23;
    Gslr_out.G32 = commonfac * n1Hn1 * factorsinc32 * factorcexp23;
    Gslr_out.G31 = commonfac * n2Hn2 * factorsinc31 * factorcexp31;
    Gslr_out.G13 = commonfac * n2Hn2 * factorsinc13 * factorcexp31;

    // ### FIXME
    // # G13 = -1j * prefactor * n2Hn2 * factorsinc31 * np.conjugate(factorcexp31)
    return Gslr_out;
}

CUDA_CALLABLE_MEMBER
d_transferL_holder d_TDICombinationFD(d_Gslr_holder Gslr, double f, int TDItag, int rescaled){
    // int TDItag == 1 is XYZ int TDItag == 2 is AET
    // int rescaled == 1 is True int rescaled == 0 is False
    d_transferL_holder transferL;
    agcmplx factor, factorAE, factorT;
    agcmplx I(0.0, 1.0);
    double x = PI*f*L_SI/C_SI;
    agcmplx z = gcmplx::exp(I*2.*x);
    agcmplx Xraw, Yraw, Zraw, Araw, Eraw, Traw;
    agcmplx factor_convention, point5, c_one, c_two;
    if (TDItag==1){
        // # First-generation TDI XYZ
        // # With x=pifL, factor scaled out: 2I*sin2x*e2ix
        if (rescaled == 1) factor = 1.;
        else factor = 2.*I*sin(2.*x)*z;
        Xraw = Gslr.G21 + z*Gslr.G12 - Gslr.G31 - z*Gslr.G13;
        Yraw = Gslr.G32 + z*Gslr.G23 - Gslr.G12 - z*Gslr.G21;
        Zraw = Gslr.G13 + z*Gslr.G31 - Gslr.G23 - z*Gslr.G32;
        transferL.transferL1 = factor * Xraw;
        transferL.transferL2 = factor * Yraw;
        transferL.transferL3 = factor * Zraw;
        return transferL;
    }

    else{
        //# First-generation TDI AET from X,Y,Z
        //# With x=pifL, factors scaled out: A,E:I*sqrt2*sin2x*e2ix T:2*sqrt2*sin2x*sinx*e3ix
        //# Here we include a factor 2, because the code was first written using the definitions (2) of McWilliams&al_0911 where A,E,T are 1/2 of their LDC definitions
        factor_convention = agcmplx(2.,0.0);
        if (rescaled == 1){
            factorAE = agcmplx(1., 0.0);
            factorT = agcmplx(1., 0.0);
        }
        else{
          factorAE = I*sqrt2*sin(2.*x)*z;
          factorT = 2.*sqrt2*sin(2.*x)*sin(x)*gcmplx::exp(I*3.*x);
        }

        Araw = 0.5 * ( (1.+z)*(Gslr.G31 + Gslr.G13) - Gslr.G23 - z*Gslr.G32 - Gslr.G21 - z*Gslr.G12 );
        Eraw = 0.5*invsqrt3 * ( (1.-z)*(Gslr.G13 - Gslr.G31) + (2.+z)*(Gslr.G12 - Gslr.G32) + (1.+2.*z)*(Gslr.G21 - Gslr.G23) );
        Traw = invsqrt6 * ( Gslr.G21 - Gslr.G12 + Gslr.G32 - Gslr.G23 + Gslr.G13 - Gslr.G31);
        transferL.transferL1 = factor_convention * factorAE * Araw;
        transferL.transferL2 = factor_convention * factorAE * Eraw;
        transferL.transferL3 = factor_convention * factorT * Traw;
        return transferL;
    }
}

CUDA_CALLABLE_MEMBER
d_transferL_holder d_JustLISAFDresponseTDI(agcmplx *H, double f, double t, double lam, double beta, double t0, int TDItag, int order_fresnel_stencil){
    t = t + t0*YRSID_SI;

    //funck
    double kvec[3] = {-cos(beta)*cos(lam), -cos(beta)*sin(lam), -sin(beta)};

    // funcp0
    double alpha = Omega0*t; double c = cos(alpha); double s = sin(alpha); double a = aorbit;
    double p0[3] = {a*c, a*s, 0.*t}; //np.array([a*c, a*s, 0.*t])

    // dot kvec with p0
    double kR = d_dot_product_1d(kvec, p0);

    double phaseRdelay = 2.*PI/clight *f*kR;

    // going to assume order_fresnel_stencil == 0 for now
    d_Gslr_holder Gslr = d_EvaluateGslr(t, f, H, kvec, 1); // assumes full response
    d_Gslr_holder Tslr; // use same struct because its the same setup
    agcmplx m_I(0.0, -1.0); // -1.0 -> mu_I

    // fill Tslr
    Tslr.G12 = Gslr.G12*gcmplx::exp(m_I*phaseRdelay); // really -I*
    Tslr.G21 = Gslr.G21*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G23 = Gslr.G23*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G32 = Gslr.G32*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G31 = Gslr.G31*gcmplx::exp(m_I*phaseRdelay);
    Tslr.G13 = Gslr.G13*gcmplx::exp(m_I*phaseRdelay);

    d_transferL_holder transferL = d_TDICombinationFD(Tslr, f, TDItag, 0);
    transferL.phaseRdelay = phaseRdelay;
    return transferL;
}


CUDA_CALLABLE_MEMBER
void kernel_JustLISAFDresponseTDI(ModeContainer *mode_vals, agcmplx *H, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points,
    double *merger_freq_arr, int TDItag, int order_fresnel_stencil, int num_walkers,
    double inc, double lam, double beta, double psi, double phi0, double t0, double tRef_wave_frame, double tRef_sampling_frame, double merger_freq,
    int mode_index, double f, int i, int walker_i
    ){
    // TDItag == 1 is XYZ, TDItag == 2 is AET
    double phasetimeshift;
    double phi_up, phi;

    double t, t_wave_frame, t_sampling_frame, x, x2, x3, coeff_0, coeff_1, coeff_2, coeff_3, f_last, Shift, t_merger, dphidf, dphidf_merger;
    int old_ind_below;

            if (i == 0) dphidf = (mode_vals[mode_index].phase[1] - mode_vals[mode_index].phase[0])/(old_freqs[walker_i*num_points + 1] - old_freqs[walker_i*num_points + 0]);
            else if(i == num_points-1) dphidf = (mode_vals[mode_index].phase[num_points-1] - mode_vals[mode_index].phase[num_points-2])/(old_freqs[walker_i*num_points + num_points-1] - old_freqs[walker_i*num_points + num_points-2]);
            else {dphidf = (mode_vals[mode_index].phase[i+1] - mode_vals[mode_index].phase[i])/(old_freqs[walker_i*num_points + i+1] - old_freqs[walker_i*num_points + i]);
            /*# if __CUDA_ARCH__>=200
                if ((i == 1183) && (walker_i == 20))
                printf("%d, %d, %d, %.12e, %.12e, %.12e, %.12e\n", walker_i, mode_i, i, mode_vals[mode_index].phase[i+1], mode_vals[mode_index].phase[i],
                old_freqs[walker_i*num_points + i+1],
                old_freqs[walker_i*num_points + i]);
            #endif //*/
        }


            /*old_ind_below = i;
            if (old_ind_below >= num_points-1) old_ind_below = num_points -2;
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            x3 = x2*x;
            coeff_0 = mode_vals[mode_index].phase[old_ind_below];
            coeff_1 = mode_vals[mode_index].phase_coeff_1[old_ind_below];
            coeff_2 = mode_vals[mode_index].phase_coeff_2[old_ind_below];
            coeff_3 = mode_vals[mode_index].phase_coeff_3[old_ind_below];

            phi = coeff_0 + coeff_1*x + 2.0*coeff_2*x2 + 3.0*coeff_3*x3;

            x = (f + 1e-6 - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            phi_up = coeff_0 + coeff_1*x + 2.0*coeff_2*x2 + 3.0*coeff_3*x3;

            dphidf = (phi_up - phi)/1e-6;*/


            /*dphidf = (mode_vals[mode_index].phase[1] - mode_vals[mode_index].phase[0])/(old_freqs[1] - old_freqs[0]);
            else if(i == num_points-1) dphidf = (mode_vals[mode_index].phase[num_points-1] - mode_vals[mode_index].phase[num_points-2])/(old_freqs[num_points-1] - old_freqs[num_points-2]);
            else dphidf = (mode_vals[mode_index].phase[i+1] - mode_vals[mode_index].phase[i])/(old_freqs[i+1] - old_freqs[i]);*/


            // Right now, it assumes same points for initial sampling of the response and the waveform
            // linear term in cubic spline is the first derivative of phase at each node point

            //dphidf = mode_vals[mode_index].phase_coeff_1[i];

            t_wave_frame = 1./(2.0*PI)*dphidf + tRef_wave_frame;
            t_sampling_frame = 1./(2.0*PI)*dphidf + tRef_sampling_frame;

            // adjust phase values stored in mode vals to reflect the tRef shift
            //mode_vals[mode_index].phase[i] += 2.0*PI*f*tRef_wave_frame;

            d_transferL_holder transferL = d_JustLISAFDresponseTDI(&H[mode_index*9], f, t_wave_frame, lam, beta, t0, TDItag, order_fresnel_stencil);

            mode_vals[mode_index].time_freq_corr[i] = t_sampling_frame + t0*YRSID_SI; // TODO: decide how to cutoff because it should be in terms of tL but it should be okay at long enough times.
            mode_vals[mode_index].transferL1_re[i] = gcmplx::real(transferL.transferL1);
            mode_vals[mode_index].transferL1_im[i] = gcmplx::imag(transferL.transferL1);
            mode_vals[mode_index].transferL2_re[i] = gcmplx::real(transferL.transferL2);
            mode_vals[mode_index].transferL2_im[i] = gcmplx::imag(transferL.transferL2);
            mode_vals[mode_index].transferL3_re[i] = gcmplx::real(transferL.transferL3);
            mode_vals[mode_index].transferL3_im[i] = gcmplx::imag(transferL.transferL3);
            mode_vals[mode_index].phaseRdelay[i] = transferL.phaseRdelay;

}

#ifdef __CUDACC__
CUDA_KERNEL
void kernel_JustLISAFDresponseTDI_wrap(ModeContainer *mode_vals, agcmplx *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double *inc_arr, double *lam_arr, double *beta_arr, double *psi_arr, double *phi0_arr, double *t0_arr, double *tRef_wave_frame_arr, double *tRef_sampling_frame_arr,
    double *merger_freq_arr, int TDItag, int order_fresnel_stencil, int num_walkers
  ){
    // TDItag == 1 is XYZ, TDItag == 2 is AET
    double inc, lam, beta, psi, phi0, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq;

    int mode_index, freq_ind;

    double f;

    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < num_walkers;
         walker_i += blockDim.z * gridDim.z){

        inc = inc_arr[walker_i];
        lam = lam_arr[walker_i];
        beta = beta_arr[walker_i];
        psi = psi_arr[walker_i];
        phi0 = phi0_arr[walker_i];
        t0 = t0_arr[walker_i];
        tRef_wave_frame = tRef_wave_frame_arr[walker_i];
        tRef_sampling_frame = tRef_sampling_frame_arr[walker_i];
        merger_freq = merger_freq_arr[walker_i];

     for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
          mode_i < num_modes;
          mode_i += blockDim.y * gridDim.y){

              mode_index = walker_i*num_modes + mode_i;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_points;
         i += blockDim.x * gridDim.x){


          freq_ind = walker_i*num_points + i;


         f = frqs[freq_ind];

           kernel_JustLISAFDresponseTDI(mode_vals, H, old_freqs, d_log10f, l_vals, m_vals, num_modes, num_points,
               merger_freq_arr, TDItag, order_fresnel_stencil, num_walkers,
               inc, lam, beta, psi, phi0, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq,
                mode_index, f, i, walker_i
              );
    }
  }
}
}

#else
void cpu_JustLISAFDresponseTDI_wrap(ModeContainer *mode_vals, agcmplx *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double *inc_arr, double *lam_arr, double *beta_arr, double *psi_arr, double *phi0_arr, double *t0_arr, double *tRef_wave_frame_arr, double *tRef_sampling_frame_arr,
    double *merger_freq_arr, int TDItag, int order_fresnel_stencil, int num_walkers
  ){
    // TDItag == 1 is XYZ, TDItag == 2 is AET
    double inc, lam, beta, psi, phi0, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq;

    int mode_index, freq_ind;

    double f;

    for (int walker_i = 0;
         walker_i < num_walkers;
         walker_i += 1){

        inc = inc_arr[walker_i];
        lam = lam_arr[walker_i];
        beta = beta_arr[walker_i];
        psi = psi_arr[walker_i];
        phi0 = phi0_arr[walker_i];
        t0 = t0_arr[walker_i];
        tRef_wave_frame = tRef_wave_frame_arr[walker_i];
        tRef_sampling_frame = tRef_sampling_frame_arr[walker_i];
        merger_freq = merger_freq_arr[walker_i];

     for (int mode_i = 0;
          mode_i < num_modes;
          mode_i += 1){

              mode_index = walker_i*num_modes + mode_i;

    for (int i = 0;
         i < num_points;
         i += 1){


          freq_ind = walker_i*num_points + i;


         f = frqs[freq_ind];

           kernel_JustLISAFDresponseTDI(mode_vals, H, old_freqs, d_log10f, l_vals, m_vals, num_modes, num_points,
               merger_freq_arr, TDItag, order_fresnel_stencil, num_walkers,
               inc, lam, beta, psi, phi0, t0, tRef_wave_frame, tRef_sampling_frame, merger_freq,
                mode_index, f, i, walker_i
              );
    }
  }
}
}
#endif


CUDA_CALLABLE_MEMBER
void kernel_add_tRef_phase_shift(ModeContainer *mode_vals, double f, int mode_index, double tRef_wave_frame, int i){
      mode_vals[mode_index].phase[i] += 2.0*PI*f*tRef_wave_frame;
}

#ifdef __CUDACC__
CUDA_KERNEL
void kernel_add_tRef_phase_shift_wrap(ModeContainer *mode_vals, double *frqs, int num_modes, int num_points, double *tRef_wave_frame_arr, int num_walkers){

    double f, tRef_wave_frame;
    int mode_index;

    for (int walker_i = blockIdx.z * blockDim.z + threadIdx.z;
         walker_i < num_walkers;
         walker_i += blockDim.z * gridDim.z){
             tRef_wave_frame = tRef_wave_frame_arr[walker_i];
     for (int mode_i = blockIdx.y * blockDim.y + threadIdx.y;
          mode_i < num_modes;
          mode_i += blockDim.y * gridDim.y){

          mode_index = walker_i*num_modes + mode_i;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x;
         i < num_points;
         i += blockDim.x * gridDim.x){

             f = frqs[walker_i*num_points + i];

            kernel_add_tRef_phase_shift(mode_vals, f, mode_index, tRef_wave_frame, i);
        }
    }
}
}

#else

void cpu_add_tRef_phase_shift_wrap(ModeContainer *mode_vals, double *frqs, int num_modes, int num_points, double *tRef_wave_frame_arr, int num_walkers){

    double f, tRef_wave_frame;
    int mode_index;

    for (int walker_i = 0;
         walker_i < num_walkers;
         walker_i += 1){
             tRef_wave_frame = tRef_wave_frame_arr[walker_i];
     for (int mode_i = 0;
          mode_i < num_modes;
          mode_i += 1){

          mode_index = walker_i*num_modes + mode_i;
    for (int i = 0;
         i < num_points;
         i += 1){

             f = frqs[walker_i*num_points + i];

            kernel_add_tRef_phase_shift(mode_vals, f, mode_index, tRef_wave_frame, i);
        }
    }
}
}
#endif



            // interpolate for time
            /*old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            if (old_ind_below >= num_points-1) old_ind_below = num_points -2;
            x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            coeff_1 = mode_vals[mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = mode_vals[mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = mode_vals[mode_i].phase_coeff_3[old_ind_below];

            t = 1./(2.*PI)*(coeff_1 + 2.0*coeff_2*x + 3.0*coeff_3*x2); // derivative of the spline

            f_last = frqs[num_points-1];
            // interpolate for time
            old_ind_below = floor((log10(f_last) - log10(old_freqs[0]))/d_log10f);
            if (old_ind_below >= num_points-1) old_ind_below = num_points -2;
            x = (f_last - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
            x2 = x*x;
            coeff_1 = mode_vals[mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = mode_vals[mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = mode_vals[mode_i].phase_coeff_3[old_ind_below];

            t_last = 1./(2.*PI)*(coeff_1 + 2.0*coeff_2*x + 3.0*coeff_3*x2); // derivative of the spline
            Shift = t_last - tc;

            if (i == 0) dphidf = (mode_vals[mode_i].phase[1] - mode_vals[mode_i].phase[0])/(old_freqs[1] - old_freqs[0]);
            else if(i == num_points-1) dphidf = (mode_vals[mode_i].phase[num_points-1] - mode_vals[mode_i].phase[num_points-2])/(old_freqs[num_points-1] - old_freqs[num_points-2]);
            else dphidf = (mode_vals[mode_i].phase[i+1] - mode_vals[mode_i].phase[i])/(old_freqs[i+1] - old_freqs[i]);
            t = -1./(2.*PI)*(dphidf); // derivative of the spline

            old_ind_below = floor((log10(merger_freq) - log10(old_freqs[0]))/d_log10f);
            if (old_ind_below >= num_points-1) old_ind_below = num_points -2;
            dphidf_merger = (mode_vals[mode_i].phase[old_ind_below+1] - mode_vals[mode_i].phase[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]); // TODO: do something more accurate?
            t_merger = -1./(2.*PI)*dphidf_merger; // derivative of the spline
            Shift = t_merger - (t0 + tRef); // TODO: think about if this is accurate for each mode

            t = t - Shift; // TODO: check */


/*
int main(){
    int num_modes = 3;
    unsigned int l[3] = {2, 3, 4};
    unsigned int m[3] = {2, 3, 4};
    double lam = 0.5;
    double beta = 0.6;
    double psi = 0.7;
    double phi0 = 0.8;
    double inc = PI/4.;

    cmplx * H = prep_H_info(&l[0], &m[0], num_modes, inc, lam, beta, psi, phi0);
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                printf("(%d, %d, %d, %d): %e, %e\n", l[mode_i], m[mode_i], i, j, cuCreal(H[mode_i*9 + i*3+j]), cuCimag(H[mode_i*9 + i*3+j]));
            }
        }
    }
    delete H;
    return (0);
}
*/
