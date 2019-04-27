#include <complex>
#include "globalPhenomHM.h"
#include <stdio.h>
#include <math.h>
#include "fdresponse.h"


using namespace std;

cmplx SpinWeightedSphericalHarmonic(int s, int l, int m, double theta, double phi){
    // l=2
    double fac;
    if ((l==2) && (m==-2)) fac =  sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 - cos( theta ))*( 1.0 - cos( theta ));
    else if ((l==2) && (m==1)) fac =  sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 - cos( theta ));
    else if ((l==2) && (m==0)) fac =  sqrt( 15.0 / ( 32.0 * PI ) ) * sin( theta )*sin( theta );
    else if ((l==2) && (m==1)) fac =  sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 + cos( theta ));
    else if ((l==2) && (m==2)) fac =  sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 + cos( theta ))*( 1.0 + cos( theta ));
    // l=3
    else if ((l==3) && (m==-3)) fac =  sqrt(21.0/(2.0*PI))*cos(theta/2.0)*pow(sin(theta/2.0),5.0);
    else if ((l==3) && (m==-2)) fac =  sqrt(7.0/4.0*PI)*(2.0 + 3.0*cos(theta))*pow(sin(theta/2.0),4.0);
    else if ((l==3) && (m==-1)) fac =  sqrt(35.0/(2.0*PI))*(sin(theta) + 4.0*sin(2.0*theta) - 3.0*sin(3.0*theta))/32.0;
    else if ((l==3) && (m==0)) fac =  (sqrt(105.0/(2.0*PI))*cos(theta)*pow(sin(theta),2.0))/4.0;
    else if ((l==3) && (m==1)) fac =  -sqrt(35.0/(2.0*PI))*(sin(theta) - 4.0*sin(2.0*theta) - 3.0*sin(3.0*theta))/32.0;
    else if ((l==3) && (m==2)) fac =  sqrt(7.0/PI)*pow(cos(theta/2.0),4.0)*(-2.0 + 3.0*cos(theta))/2.0;
    else if ((l==3) && (m==3)) fac =  -sqrt(21.0/(2.0*PI))*pow(cos(theta/2.0),5.0)*sin(theta/2.0);
    // l=4
    else if ((l==4) && (m==-4)) fac =  3.0*sqrt(7.0/PI)*pow(cos(theta/2.0),2.0)*pow(sin(theta/2.0),6.0);
    else if ((l==4) && (m==-3)) fac =  3.0*sqrt(7.0/(2.0*PI))*cos(theta/2.0)*(1.0 + 2.0*cos(theta))*pow(sin(theta/2.0),5.0);
    else if ((l==4) && (m==-2)) fac =  (3.0*(9.0 + 14.0*cos(theta) + 7.0*cos(2.0*theta))*pow(sin(theta/2.0),4.0))/(4.0*sqrt(PI));
    else if ((l==4) && (m==-1)) fac =  (3.0*(3.0*sin(theta) + 2.0*sin(2.0*theta) + 7.0*sin(3.0*theta) - 7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*PI));
    else if ((l==4) && (m==0)) fac =  (3.0*sqrt(5.0/(2.0*PI))*(5.0 + 7.0*cos(2.0*theta))*pow(sin(theta),2.0))/16.0;
    else if ((l==4) && (m==1)) fac =  (3.0*(3.0*sin(theta) - 2.0*sin(2.0*theta) + 7.0*sin(3.0*theta) + 7.0*sin(4.0*theta)))/(32.0*sqrt(2.0*PI));
    else if ((l==4) && (m==2)) fac =  (3.0*pow(cos(theta/2.0),4.0)*(9.0 - 14.0*cos(theta) + 7.0*cos(2.0*theta)))/(4.0*sqrt(PI));
    else if ((l==4) && (m==3)) fac =  -3.0*sqrt(7.0/(2.0*PI))*pow(cos(theta/2.0),5.0)*(-1.0 + 2.0*cos(theta))*sin(theta/2.0);
    else if ((l==4) && (m==4)) fac =  3.0*sqrt(7.0/PI)*pow(cos(theta/2.0),6.0)*pow(sin(theta/2.0),2.0);
    // l= 5
    else if ((l==5) && (m==-5)) fac =  sqrt(330.0/PI)*pow(cos(theta/2.0),3.0)*pow(sin(theta/2.0),7.0);
    else if ((l==5) && (m==-4)) fac =  sqrt(33.0/PI)*pow(cos(theta/2.0),2.0)*(2.0 + 5.0*cos(theta))*pow(sin(theta/2.0),6.0);
    else if ((l==5) && (m==-3)) fac =  (sqrt(33.0/(2.0*PI))*cos(theta/2.0)*(17.0 + 24.0*cos(theta) + 15.0*cos(2.0*theta))*pow(sin(theta/2.0),5.0))/4.0;
    else if ((l==5) && (m==-2)) fac =  (sqrt(11.0/PI)*(32.0 + 57.0*cos(theta) + 36.0*cos(2.0*theta) + 15.0*cos(3.0*theta))*pow(sin(theta/2.0),4.0))/8.0;
    else if ((l==5) && (m==-1)) fac =  (sqrt(77.0/PI)*(2.0*sin(theta) + 8.0*sin(2.0*theta) + 3.0*sin(3.0*theta) + 12.0*sin(4.0*theta) - 15.0*sin(5.0*theta)))/256.0;
    else if ((l==5) && (m==0)) fac =  (sqrt(1155.0/(2.0*PI))*(5.0*cos(theta) + 3.0*cos(3.0*theta))*pow(sin(theta),2.0))/32.0;
    else if ((l==5) && (m==1)) fac =  sqrt(77.0/PI)*(-2.0*sin(theta) + 8.0*sin(2.0*theta) - 3.0*sin(3.0*theta) + 12.0*sin(4.0*theta) + 15.0*sin(5.0*theta))/256.0;
    else if ((l==5) && (m==2)) fac =  sqrt(11.0/PI)*pow(cos(theta/2.0),4.0)*(-32.0 + 57.0*cos(theta) - 36.0*cos(2.0*theta) + 15.0*cos(3.0*theta))/8.0;
    else if ((l==5) && (m==3)) fac =  -sqrt(33.0/(2.0*PI))*pow(cos(theta/2.0),5.0)*(17.0 - 24.0*cos(theta) + 15.0*cos(2.0*theta))*sin(theta/2.0)/4.0;
    else if ((l==5) && (m==4)) fac =  sqrt(33.0/PI)*pow(cos(theta/2.0),6.0)*(-2.0 + 5.0*cos(theta))*pow(sin(theta/2.0),2.0);
    else if ((l==5) && (m==5)) fac =  -sqrt(330.0/PI)*pow(cos(theta/2.0),7.0)*pow(sin(theta/2.0),3.0);
    else printf("Spherical harmonic for %d, %d not implemented\n", l, m); // TODO: add error

    // Result
    cmplx I(0.0, 1.0);
    if (m==0) return cmplx(fac, 0.0);
    else {
        cmplx phaseTerm(m*phi, 0.0);
        return fac * exp(I*phaseTerm);
    }
}

double sinc(double x){
    if (x == 0.0) return 1.0;
    else return sin(x)/x;
}


void dot_product_2d(double out[3][3], double arr1[3][3], int m1, int n1, double arr2[3][3], int m2, int n2){
    for (int i=0; i<m1; i++){
        for (int j=0; j<n2; j++){
            for (int k=0; k<n1; k++){
                out[i][j] += arr1[i][k]*arr2[k][j];
            }
        }
    }
}

double dot_product_1d(double arr1[3], double arr2[3]){
    double out = 0.0;
    for (int i=0; i<3; i++){
        out += arr1[i]*arr2[i];
    }
    return out;
}

cmplx vec_H_vec_product(double arr1[3], cmplx *H, double arr2[3]){
    cmplx I(0.0, 1.0);
    cmplx out(0.0, 0.0);
    cmplx trans;
    for (int i=0; i<3; i++){
        trans = 0.0 * I*0.0;
        for (int j=0; j<3; j++){
            trans += H[i*3 + j]*arr2[j];
        }
        out += arr1[i]*trans;
    }
    return out;
}

cmplx * prep_H_info(unsigned int *l_vals, unsigned int *m_vals, int num_modes, double inc, double lam, double beta, double psi, double phi0){

    //##### Based on the f-n by Sylvain   #####
    double HSplus[3][3] =
    {
        {1., 0., 0.},
        {0., -1., 0.},
        {0., 0., 0.}
    };

    double HScross[3][3] =
    {
        {0., 1., 0.},
        {1., 0., 0.},
        {0., 0., 0.}
    };

    // Wave unit vector
    double kvec[3] = {-cos(beta)*cos(lam), -cos(beta)*sin(lam), -sin(beta)};

    // Compute constant matrices Hplus and Hcross in the SSB frame
    double clambd = cos(lam); double slambd = sin(lam);
    double cbeta = cos(beta); double sbeta = sin(beta);
    double cpsi = cos(psi); double spsi = sin(psi);

    double O1[3][3] =
    {
        {cpsi*slambd-clambd*sbeta*spsi,-clambd*cpsi*sbeta-slambd*spsi,-cbeta*clambd},
        {-clambd*cpsi-sbeta*slambd*spsi,-cpsi*sbeta*slambd+clambd*spsi,-cbeta*slambd},
        {cbeta*spsi,cbeta*cpsi,-sbeta}
    };

    double invO1[3][3] =
    {
        {cpsi*slambd-clambd*sbeta*spsi,-clambd*cpsi-sbeta*slambd*spsi,cbeta*spsi},
        {-clambd*cpsi*sbeta-slambd*spsi,-cpsi*sbeta*slambd+clambd*spsi,cbeta*cpsi},
        {-cbeta*clambd,-cbeta*slambd,-sbeta}
    };
    double out1[3][3] = {0};
    double out2[3][3] = {0};

    double Hplus[3][3] = {0};
    double Hcross[3][3] = {0};

    // get Hplus
    dot_product_2d(out1, HSplus, 3, 3, invO1, 3, 3);
    dot_product_2d(Hplus, O1, 3, 3, out1, 3, 3);

    // get Hcross
    dot_product_2d(out2, HScross, 3, 3, invO1, 3, 3);
    dot_product_2d(Hcross, O1, 3, 3, out2, 3, 3);

    cmplx I(0.0, 1.0);
    cmplx Ylm, Yl_m, Yfactorplus, Yfactorcross;
    int dim = 9*num_modes;
    cmplx *H = new cmplx[dim];
    cmplx trans1, trans2;
    int m;
    int l;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        l = l_vals[mode_i];
        m = m_vals[mode_i];
        Ylm = SpinWeightedSphericalHarmonic(-2, l, m, inc, phi0);
        Yl_m = std::conj(SpinWeightedSphericalHarmonic(-2, l, -1*m, inc, phi0));
        Yfactorplus = 1./2 * (Ylm + Yl_m);
        //# Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, should be for correct phase conventions
        Yfactorcross = 1./2. * I * (Ylm - Yl_m); //  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# Yfactorcross = -1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# The matrix H is now complex

        //# H = np.conjugate((Yfactorplus*Hplus + Yfactorcross*Hcross))  ### SB: H_ij = H A_22 exp(i\Psi(f))
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                trans1 = Hplus[i][j];
                trans2 = Hcross[i][j];
                H[mode_i*9 + i*3 + j] = (Yfactorplus*trans1+ Yfactorcross*trans2);
                //printf("(%d, %d): %e, %e\n", i, j, Hplus[i][j], Hcross[i][j]);
            }
        }
    }
    return &H[0];
}

/* # Single-link response
# 'full' does include the orbital-delay term, 'constellation' does not
# t can be a scalar or a 1D vector */
Gslr_holder EvaluateGslr(double t, double f, cmplx *H, double k[3], int response){
    // response == 1 is full ,, response anything else is constellation
    //# Trajectories, p0 used only for the full response
    cmplx I(0.0, 1.0);
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
    double kn1 = dot_product_1d(k, n1);
    double kn2 = dot_product_1d(k, n2);
    double kn3 = dot_product_1d(k, n3);

    cmplx n1Hn1 = vec_H_vec_product(n1, H, n1); //np.dot(n1, np.dot(H, n1))
    cmplx n2Hn2 = vec_H_vec_product(n2, H, n2); //np.dot(n2, np.dot(H, n2))
    cmplx n3Hn3 = vec_H_vec_product(n3, H, n3); //np.dot(n3, np.dot(H, n3))

    double p1L_plus_p2L[3] = {p1L[0]+p2L[0], p1L[1]+p2L[1], p1L[2]+p2L[2]};
    double p2L_plus_p3L[3] = {p2L[0]+p3L[0], p2L[1]+p3L[1], p2L[2]+p3L[2]};
    double p3L_plus_p1L[3] = {p3L[0]+p1L[0], p3L[1]+p1L[1], p3L[2]+p1L[2]};

    double kp1Lp2L = dot_product_1d(k, p1L_plus_p2L);
    double kp2Lp3L = dot_product_1d(k, p2L_plus_p3L);
    double kp3Lp1L = dot_product_1d(k, p3L_plus_p1L);
    double kp0 = dot_product_1d(k, p0);

    // # Prefactors - projections are either scalars or vectors
    cmplx factorcexp0;
    if (response==1) factorcexp0 = exp(I*2.*PI*f/C_SI * kp0);
    else factorcexp0 = 1.;
    double prefactor = PI*f*L_SI/C_SI;

    cmplx factorcexp12 = exp(I*prefactor * (1.+kp1Lp2L/L_SI));
    cmplx factorcexp23 = exp(I*prefactor * (1.+kp2Lp3L/L_SI));
    cmplx factorcexp31 = exp(I*prefactor * (1.+kp3Lp1L/L_SI));

    cmplx factorsinc12 = sinc( prefactor * (1.-kn3));
    cmplx factorsinc21 = sinc( prefactor * (1.+kn3));
    cmplx factorsinc23 = sinc( prefactor * (1.-kn1));
    cmplx factorsinc32 = sinc( prefactor * (1.+kn1));
    cmplx factorsinc31 = sinc( prefactor * (1.-kn2));
    cmplx factorsinc13 = sinc( prefactor * (1.+kn2));

    // # Compute the Gslr - either scalars or vectors
    Gslr_holder Gslr_out;

    cmplx commonfac = I * prefactor * factorcexp0;
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

transferL_holder TDICombinationFD(Gslr_holder Gslr, double f, int TDItag, int rescaled){
    // int TDItag == 1 is XYZ int TDItag == 2 is AET
    // int rescaled == 1 is True int rescaled == 0 is False
    transferL_holder transferL;
    cmplx factor, factorAE, factorT;
    cmplx I(0.0, 1.0);
    double x = PI*f*L_SI/C_SI;
    cmplx z = exp(2.*I*x);
    cmplx Xraw, Yraw, Zraw, Araw, Eraw, Traw;
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
        double factor_convention = 2.;
        if (rescaled == 1){
            factorAE = 1.;
            factorT = 1.;
        }
        else{
            factorAE = I*sqrt2*sin(2.*x)*z;
            factorT = 2.*sqrt2*sin(2.*x)*sin(x)*exp(I*3.*x);
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

transferL_holder JustLISAFDresponseTDI(cmplx *H, double f, double t, double lam, double beta, double t0, int TDItag, int order_fresnel_stencil){
    t = t + t0*YRSID_SI;

    //funck
    double kvec[3] = {-cos(beta)*cos(lam), -cos(beta)*sin(lam), -sin(beta)};

    // funcp0
    double alpha = Omega0*t; double c = cos(alpha); double s = sin(alpha); double a = aorbit;
    double p0[3] = {a*c, a*s, 0.*t}; //np.array([a*c, a*s, 0.*t])

    // dot kvec with p0
    double kR = dot_product_1d(kvec, p0);

    double phaseRdelay = 2.*PI/clight *f*kR;

    // going to assume order_fresnel_stencil == 0 for now
    Gslr_holder Gslr = EvaluateGslr(t, f, H, kvec, 1); // assumes full response
    Gslr_holder Tslr; // use same struct because its the same setup
    cmplx I(0.0, 1.0);

    // fill Tslr
    Tslr.G12 = Gslr.G12 * exp(-I*phaseRdelay);
    Tslr.G21 = Gslr.G21 * exp(-I*phaseRdelay);
    Tslr.G23 = Gslr.G23 * exp(-I*phaseRdelay);
    Tslr.G32 = Gslr.G32 * exp(-I*phaseRdelay);
    Tslr.G31 = Gslr.G31 * exp(-I*phaseRdelay);
    Tslr.G13 = Gslr.G13 * exp(-I*phaseRdelay);

    transferL_holder transferL = TDICombinationFD(Tslr, f, TDItag, 0);
    transferL.phaseRdelay = phaseRdelay;
    return transferL;
}



void JustLISAFDresponseTDI_wrap(ModeContainer *mode_vals, cmplx *H, double *frqs, double *old_freqs, double d_log10f, unsigned int *l_vals, unsigned int *m_vals, int num_modes, int num_points, double inc, double lam, double beta, double psi, double phi0, double tc, double tShift, int TDItag, int order_fresnel_stencil){
    // TDItag == 1 is XYZ, TDItag == 2 is AET
    double t0 = 0.0;
    double phasetimeshift;
    double f, t, x, x2, coeff_1, coeff_2, coeff_3;
    int old_ind_below;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        for (int i=0; i<num_points; i++){
            f = frqs[i];
            // interpolate for time
            old_ind_below = floor((log10(f) - log10(old_freqs[0]))/d_log10f);
            x = f - old_freqs[old_ind_below];
            x2 = x*x;
            coeff_1 = mode_vals[mode_i].phase_coeff_1[old_ind_below];
            coeff_2 = mode_vals[mode_i].phase_coeff_2[old_ind_below];
            coeff_3 = mode_vals[mode_i].phase_coeff_3[old_ind_below];

            t = 1./(2.*PI)*(coeff_1 + 2.0*coeff_2*x + 3.0*coeff_3*x2); // derivative of the spline
            transferL_holder transferL = JustLISAFDresponseTDI(&H[mode_i*9], f, t, lam, beta, t0, TDItag, order_fresnel_stencil);

            mode_vals[mode_i].transferL1_re[i] = std::real(transferL.transferL1);
            mode_vals[mode_i].transferL1_im[i] = std::imag(transferL.transferL1);
            mode_vals[mode_i].transferL2_re[i] = std::real(transferL.transferL2);
            mode_vals[mode_i].transferL2_im[i] = std::imag(transferL.transferL2);
            mode_vals[mode_i].transferL3_re[i] = std::real(transferL.transferL3);
            mode_vals[mode_i].transferL3_im[i] = std::imag(transferL.transferL3);
            mode_vals[mode_i].phaseRdelay[i] = transferL.phaseRdelay;
        }
    }
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
                printf("(%d, %d, %d, %d): %e, %e\n", l[mode_i], m[mode_i], i, j, std::real(H[mode_i*9 + i*3+j]), std::imag(H[mode_i*9 + i*3+j]));
            }
        }
    }
    delete H;
    return (0);
}
*/
