#include <cuComplex.h>
#include "globalPhenomHM.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "fdresponse.h"

using namespace std;

typedef struct tagd_Gslr_holder{
    cuDoubleComplex G21;
    cuDoubleComplex G12;
    cuDoubleComplex G23;
    cuDoubleComplex G32;
    cuDoubleComplex G31;
    cuDoubleComplex G13;
} d_Gslr_holder;

typedef struct tagd_transferL_holder{
    cuDoubleComplex transferL1;
    cuDoubleComplex transferL2;
    cuDoubleComplex transferL3;
    double phaseRdelay;
} d_transferL_holder;

__device__
double d_sinc(double x){
    if (x == 0.0) return 1.0;
    else return sin(x)/x;
}

__device__
cuDoubleComplex complex_exp (cuDoubleComplex arg)
{
   cuDoubleComplex res;
   double s, c;
   double e = exp(arg.x);
   sincos(arg.y, &s, &c);
   res.x = c * e;
   res.y = s * e;
   return res;
}


__device__
double d_dot_product_1d(double arr1[3], double arr2[3]){
    double out = 0.0;
    for (int i=0; i<3; i++){
        out += arr1[i]*arr2[i];
    }
    return out;
}

__device__
cuDoubleComplex d_vec_H_vec_product(double arr1[3], cuDoubleComplex *H, double arr2[3]){
    cuDoubleComplex c_arr1[3] = {make_cuDoubleComplex(arr1[0], 0.0),
                                 make_cuDoubleComplex(arr1[1], 0.0),
                                 make_cuDoubleComplex(arr1[2], 0.0)};

    cuDoubleComplex c_arr2[3] = {make_cuDoubleComplex(arr2[0], 0.0),
                                 make_cuDoubleComplex(arr2[1], 0.0),
                                 make_cuDoubleComplex(arr2[2], 0.0)};

    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    cuDoubleComplex out = make_cuDoubleComplex(0.0, 0.0);
    cuDoubleComplex trans;
    for (int i=0; i<3; i++){
        trans = make_cuDoubleComplex(0.0, 0.0);
        for (int j=0; j<3; j++){
            trans = cuCadd(trans, cuCmul(H[i*3 + j], c_arr2[j]));
        }
        out = cuCadd(out, cuCmul(c_arr1[i], trans));
    }
    return out;
}

/* # Single-link response
# 'full' does include the orbital-delay term, 'constellation' does not
# t can be a scalar or a 1D vector */
__device__
d_Gslr_holder d_EvaluateGslr(double t, double f, cuDoubleComplex *H, double k[3], int response){
    // response == 1 is full ,, response anything else is constellation
    //# Trajectories, p0 used only for the full response
    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    cuDoubleComplex m_I = make_cuDoubleComplex(0.0, -1.0);
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

    cuDoubleComplex n1Hn1 = d_vec_H_vec_product(n1, H, n1); //np.dot(n1, np.dot(H, n1))
    cuDoubleComplex n2Hn2 = d_vec_H_vec_product(n2, H, n2); //np.dot(n2, np.dot(H, n2))
    cuDoubleComplex n3Hn3 = d_vec_H_vec_product(n3, H, n3); //np.dot(n3, np.dot(H, n3))

    double p1L_plus_p2L[3] = {p1L[0]+p2L[0], p1L[1]+p2L[1], p1L[2]+p2L[2]};
    double p2L_plus_p3L[3] = {p2L[0]+p3L[0], p2L[1]+p3L[1], p2L[2]+p3L[2]};
    double p3L_plus_p1L[3] = {p3L[0]+p1L[0], p3L[1]+p1L[1], p3L[2]+p1L[2]};

    double kp1Lp2L = d_dot_product_1d(k, p1L_plus_p2L);
    double kp2Lp3L = d_dot_product_1d(k, p2L_plus_p3L);
    double kp3Lp1L = d_dot_product_1d(k, p3L_plus_p1L);
    double kp0 = d_dot_product_1d(k, p0);

    // # Prefactors - projections are either scalars or vectors
    cuDoubleComplex factorccuCexp0;
    if (response==1) factorccuCexp0 = complex_exp(cuCmul(I, make_cuDoubleComplex(2.*PI*f/C_SI * kp0, 0.0))); // I*2.*PI*f/C_SI * kp0
    else factorccuCexp0 = make_cuDoubleComplex(1.0, 0.0);
    double prefactor = PI*f*L_SI/C_SI;

    cuDoubleComplex factorccuCexp12 = complex_exp(cuCmul(I, make_cuDoubleComplex(prefactor * (1.+kp1Lp2L/L_SI), 0.0))); //prefactor * (1.+kp1Lp2L/L_SI)
    cuDoubleComplex factorccuCexp23 = complex_exp(cuCmul(I, make_cuDoubleComplex(prefactor * (1.+kp2Lp3L/L_SI), 0.0))); //prefactor * (1.+kp2Lp3L/L_SI)
    cuDoubleComplex factorccuCexp31 = complex_exp(cuCmul(I, make_cuDoubleComplex(prefactor * (1.+kp3Lp1L/L_SI), 0.0))); //prefactor * (1.+kp3Lp1L/L_SI)

    cuDoubleComplex factorsinc12 = make_cuDoubleComplex(d_sinc( prefactor * (1.-kn3)),0.0);
    cuDoubleComplex factorsinc21 = make_cuDoubleComplex(d_sinc( prefactor * (1.+kn3)),0.0);
    cuDoubleComplex factorsinc23 = make_cuDoubleComplex(d_sinc( prefactor * (1.-kn1)),0.0);
    cuDoubleComplex factorsinc32 = make_cuDoubleComplex(d_sinc( prefactor * (1.+kn1)),0.0);
    cuDoubleComplex factorsinc31 = make_cuDoubleComplex(d_sinc( prefactor * (1.-kn2)),0.0);
    cuDoubleComplex factorsinc13 = make_cuDoubleComplex(d_sinc( prefactor * (1.+kn2)),0.0);

    // # Compute the Gslr - either scalars or vectors
    d_Gslr_holder Gslr_out;

    cuDoubleComplex commonfac = cuCmul(cuCmul(I, make_cuDoubleComplex(prefactor, 0.0)), factorccuCexp0);
    Gslr_out.G12 = cuCmul(cuCmul(cuCmul(commonfac, n3Hn3), factorsinc12), factorccuCexp12);
    Gslr_out.G21 = cuCmul(cuCmul(cuCmul(commonfac, n3Hn3), factorsinc21), factorccuCexp12);
    Gslr_out.G23 = cuCmul(cuCmul(cuCmul(commonfac, n1Hn1), factorsinc23), factorccuCexp23);
    Gslr_out.G32 = cuCmul(cuCmul(cuCmul(commonfac, n1Hn1), factorsinc32), factorccuCexp23);
    Gslr_out.G31 = cuCmul(cuCmul(cuCmul(commonfac, n2Hn2), factorsinc31), factorccuCexp31);
    Gslr_out.G13 = cuCmul(cuCmul(cuCmul(commonfac, n2Hn2), factorsinc13), factorccuCexp31);

    // ### FIXME
    // # G13 = -1j * prefactor * n2Hn2 * factorsinc31 * np.conjugate(factorccuCexp31)
    return Gslr_out;
}

__device__
d_transferL_holder d_TDICombinationFD(d_Gslr_holder Gslr, double f, int TDItag, int rescaled){
    // int TDItag == 1 is XYZ int TDItag == 2 is AET
    // int rescaled == 1 is True int rescaled == 0 is False
    d_transferL_holder transferL;
    cuDoubleComplex factor, factorAE, factorT;
    cuDoubleComplex I = make_cuDoubleComplex(0.0, 1.0);
    double x = PI*f*L_SI/C_SI;
    cuDoubleComplex z = complex_exp(cuCmul(I, make_cuDoubleComplex(2.*x, 0.0)));
    cuDoubleComplex Xraw, Yraw, Zraw, Araw, Eraw, Traw;
    cuDoubleComplex factor_convention, point5, c_one, c_two;
    if (TDItag==1){
        // # First-generation TDI XYZ
        // # With x=pifL, factor scaled out: 2I*sin2x*e2ix
        if (rescaled == 1) factor = make_cuDoubleComplex(1., 0.0);
        else factor = cuCmul(I, cuCmul(z, make_cuDoubleComplex(2.*sin(2.*x), 0.0)));
        Xraw = cuCsub(cuCadd(Gslr.G21, cuCmul(z, Gslr.G12)), cuCadd(Gslr.G31,cuCmul(z, Gslr.G13)));
        Yraw = cuCsub(cuCadd(Gslr.G32, cuCmul(z,Gslr.G23)), cuCadd(Gslr.G12, cuCmul(z,Gslr.G21)));
        Zraw = cuCsub(cuCadd(Gslr.G13, cuCmul(z,Gslr.G31)), cuCadd(Gslr.G23, cuCmul(z,Gslr.G32)));
        transferL.transferL1 = cuCmul(factor, Xraw);
        transferL.transferL2 = cuCmul(factor, Yraw);
        transferL.transferL3 = cuCmul(factor, Zraw);
        return transferL;
    }

    else{
        //# First-generation TDI AET from X,Y,Z
        //# With x=pifL, factors scaled out: A,E:I*sqrt2*sin2x*e2ix T:2*sqrt2*sin2x*sinx*e3ix
        //# Here we include a factor 2, because the code was first written using the definitions (2) of McWilliams&al_0911 where A,E,T are 1/2 of their LDC definitions
        factor_convention = make_cuDoubleComplex(2.,0.0);
        if (rescaled == 1){
            factorAE = make_cuDoubleComplex(1., 0.0);
            factorT = make_cuDoubleComplex(1., 0.0);
        }
        else{
            factorAE = cuCmul(I, cuCmul(z, make_cuDoubleComplex(sqrt2*sin(2.*x), 0.0)));
            factorT = cuCmul(make_cuDoubleComplex(2.*sqrt2*sin(2.*x)*sin(x), 0.0), complex_exp(cuCmul(I, make_cuDoubleComplex(3.*x, 0.0))));
        }

        point5 = make_cuDoubleComplex(0.5, 0.0);
        c_one = make_cuDoubleComplex(1.0, 0.0);
        c_two = make_cuDoubleComplex(2.0, 0.0);
        Araw = cuCmul(point5, cuCsub(cuCmul(cuCadd(c_one, z),cuCadd(Gslr.G31, Gslr.G13)), cuCadd(cuCadd(Gslr.G23, cuCmul(z, Gslr.G32)), cuCadd(Gslr.G21, cuCmul(z, Gslr.G12 )))));

        Eraw = cuCmul(cuCmul(point5, make_cuDoubleComplex(invsqrt3, 0.0)), cuCadd(cuCmul(cuCsub(c_one, z), cuCsub(Gslr.G13, Gslr.G31)), cuCadd(cuCmul(cuCadd(c_two,z),cuCsub(Gslr.G12, Gslr.G32)), cuCmul(cuCadd(c_one, cuCmul(c_two,z)), cuCsub(Gslr.G21, Gslr.G23)))));

        Traw = cuCmul(make_cuDoubleComplex(invsqrt6, 0.0), cuCadd(cuCsub(Gslr.G21, Gslr.G12), cuCadd(cuCsub(Gslr.G32, Gslr.G23), cuCsub(Gslr.G13, Gslr.G31))));

        transferL.transferL1 = cuCmul(cuCmul(factor_convention, factorAE), Araw);
        transferL.transferL2 = cuCmul(cuCmul(factor_convention, factorAE), Eraw);
        transferL.transferL3 = cuCmul(cuCmul(factor_convention, factorT), Traw);
        return transferL;
    }
}

__device__
d_transferL_holder d_JustLISAFDresponseTDI(cuDoubleComplex *H, double f, double t, double lam, double beta, double t0, int TDItag, int order_fresnel_stencil){
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
    cuDoubleComplex m_I = make_cuDoubleComplex(0.0, -1.0); // -1.0 -> mu_I

    // fill Tslr
    cuDoubleComplex complex_phaseRdelay = make_cuDoubleComplex(phaseRdelay, 0.0);
    Tslr.G12 = cuCmul(Gslr.G12, complex_exp(cuCmul(m_I,complex_phaseRdelay))); // really -I*
    Tslr.G21 = cuCmul(Gslr.G21, complex_exp(cuCmul(m_I,complex_phaseRdelay)));
    Tslr.G23 = cuCmul(Gslr.G23, complex_exp(cuCmul(m_I,complex_phaseRdelay)));
    Tslr.G32 = cuCmul(Gslr.G32, complex_exp(cuCmul(m_I,complex_phaseRdelay)));
    Tslr.G31 = cuCmul(Gslr.G31, complex_exp(cuCmul(m_I,complex_phaseRdelay)));
    Tslr.G13 = cuCmul(Gslr.G13, complex_exp(cuCmul(m_I,complex_phaseRdelay)));

    d_transferL_holder transferL = d_TDICombinationFD(Tslr, f, TDItag, 0);
    transferL.phaseRdelay = phaseRdelay;
    return transferL;
}


__global__
void kernel_JustLISAFDresponseTDI_wrap(ModeContainer *mode_vals, cuDoubleComplex *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double inc, double lam, double beta, double psi, double phi0, double t0, double tRef, double merger_freq, int TDItag, int order_fresnel_stencil, PhenomHMStorage *pHM){
    // TDItag == 1 is XYZ, TDItag == 2 is AET
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    int mode_i = blockIdx.y;
    if (i>=num_points) return;
    if (mode_i >= num_modes) return;
    double phasetimeshift;
    double f, t, x, x2, coeff_1, coeff_2, coeff_3, f_last, Shift, t_merger, dphidf, dphidf_merger, merger_freq_lm;
    int old_ind_below, ell, mm;

            f = frqs[i];
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
            Shift = t_last - tc;*/

            if (i == 0) dphidf = (mode_vals[mode_i].phase[1] - mode_vals[mode_i].phase[0])/(old_freqs[1] - old_freqs[0]);
            else if(i == num_points-1) dphidf = (mode_vals[mode_i].phase[num_points-1] - mode_vals[mode_i].phase[num_points-2])/(old_freqs[num_points-1] - old_freqs[num_points-2]);
            else dphidf = (mode_vals[mode_i].phase[i+1] - mode_vals[mode_i].phase[i])/(old_freqs[i+1] - old_freqs[i]);
            t = 1./(2.*PI)*(dphidf); // derivative of the spline

            ell = mode_vals[mode_i].l;
            mm = mode_vals[mode_i].m;

            // Shift merger frequency for each mode to get accurate tRef
            merger_freq_lm = merger_freq * (pHM->PhenomHMfring[ell][mm]/pHM->PhenomHMfring[2][2]);
            old_ind_below = floor((log10(merger_freq_lm) - log10(old_freqs[0]))/d_log10f);
            if (old_ind_below >= num_points-1) old_ind_below = num_points -2;
            dphidf_merger = (mode_vals[mode_i].phase[old_ind_below+1] - mode_vals[mode_i].phase[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]); // TODO: do something more accurate?
            t_merger = 1./(2.*PI)*dphidf_merger; // derivative of the spline*/
            Shift = t_merger - (t0 + tRef); // TODO: think about if this is accurate for each mode

            t = t - Shift;

            d_transferL_holder transferL = d_JustLISAFDresponseTDI(&H[mode_i*9], f, t, lam, beta, t0, TDItag, order_fresnel_stencil);

            mode_vals[mode_i].time_freq_corr[i] = t;
            mode_vals[mode_i].transferL1_re[i] = cuCreal(transferL.transferL1);
            mode_vals[mode_i].transferL1_im[i] = cuCimag(transferL.transferL1);
            mode_vals[mode_i].transferL2_re[i] = cuCreal(transferL.transferL2);
            mode_vals[mode_i].transferL2_im[i] = cuCimag(transferL.transferL2);
            mode_vals[mode_i].transferL3_re[i] = cuCreal(transferL.transferL3);
            mode_vals[mode_i].transferL3_im[i] = cuCimag(transferL.transferL3);
            mode_vals[mode_i].phaseRdelay[i] = transferL.phaseRdelay;
}

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
