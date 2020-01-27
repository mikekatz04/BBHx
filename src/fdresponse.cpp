/*  This code was created by Michael Katz.
 *  It is shared under the GNU license (see below).
 *  This code computes the fast Fourier domain response function for LISA
 *  based on Marsat and Baker 2018. This code contains the CPU side to mirror GPU functions
 *  for this calculation.
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

#include <complex>
#include "globalPhenomHM.h"
#include <stdio.h>
#include <math.h>
#include "fdresponse.h"


using namespace std;

/*
Calculate spin weighted spherical harmonics
*/
cmplx SpinWeightedSphericalHarmonic(int s, int l, int m, double theta, double phi){
    // l=2
    double fac;
    if ((l==2) && (m==-2)) fac =  sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 - cos( theta ))*( 1.0 - cos( theta ));
    else if ((l==2) && (m==-1)) fac =  sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 - cos( theta ));
    else if ((l==2) && (m==0)) fac =  sqrt( 15.0 / ( 32.0 * PI ) ) * sin( theta )*sin( theta );
    else if ((l==2) && (m==1)) fac =  sqrt( 5.0 / ( 16.0 * PI ) ) * sin( theta )*( 1.0 + cos( theta ));
    else if ((l==2) && (m==2)) fac =  sqrt( 5.0 / ( 64.0 * PI ) ) * ( 1.0 + cos( theta ))*( 1.0 + cos( theta ));
    // l=3
    else if ((l==3) && (m==-3)) fac =  sqrt(21.0/(2.0*PI))*cos(theta/2.0)*pow(sin(theta/2.0),5.0);
    else if ((l==3) && (m==-2)) fac =  sqrt(7.0/(4.0*PI))*(2.0 + 3.0*cos(theta))*pow(sin(theta/2.0),4.0);
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

/*
custom sinc function
*/
double sinc(double x){
    if (x == 0.0) return 1.0;
    else return sin(x)/x;
}

/*
custom dot product in 2d
*/
void dot_product_2d(double out[3][3], double arr1[3][3], int m1, int n1, double arr2[3][3], int m2, int n2){
    for (int i=0; i<m1; i++){
        for (int j=0; j<n2; j++){
            for (int k=0; k<n1; k++){
                out[i][j] += arr1[i][k]*arr2[k][j];
            }
        }
    }
}

/*
Custom dot product in 1d
*/
double dot_product_1d(double arr1[3], double arr2[3]){
    double out = 0.0;
    for (int i=0; i<3; i++){
        out += arr1[i]*arr2[i];
    }
    return out;
}

/*
Function for calculating matrix calculations of vectors with H matrix
*/
cmplx vec_H_vec_product(double arr1[3], cmplx *H_mat, double arr2[3]){
    cmplx I(0.0, 1.0);
    cmplx out(0.0, 0.0);
    cmplx trans;
    for (int i=0; i<3; i++){
        trans = 0.0 * I*0.0;
        for (int j=0; j<3; j++){
            trans += H_mat[i*3 + j]*arr2[j];
        }
        out += arr1[i]*trans;
    }
    return out;
}

/*
Get H matrix for projections based on source location.
*/
void prep_H_info(cmplx *H_mat, unsigned int *l_vals, unsigned int *m_vals, int num_modes, double inc, double lam, double beta, double psi, double phi0){

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

    cmplx I = cmplx(0.0, 1.0);
    cmplx Ylm, Yl_m, Yfactorplus, Yfactorcross;

    cmplx trans1, trans2;
    int m;
    int l;
    for (int mode_i=0; mode_i<num_modes; mode_i++){
        l = l_vals[mode_i];
        m = m_vals[mode_i];
        Ylm = SpinWeightedSphericalHarmonic(-2, l, m, inc, phi0);
        Yl_m = pow(-1.0, l)*std::conj(SpinWeightedSphericalHarmonic(-2, l, -1*m, inc, phi0));
        Yfactorplus = 1./2 * (Ylm + Yl_m);
        //# Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, should be for correct phase conventions
        Yfactorcross = 1./2. * I * (Ylm - Yl_m); //  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# Yfactorcross = -1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# Yfactorcross = 1j/2 * (Y22 - Y2m2)  ### SB, minus because the phase convention is opposite, we'll tace c.c. at the end
        //# The matrix H_mat is now complex

        //# H_mat = np.conjugate((Yfactorplus*Hplus + Yfactorcross*Hcross))  ### SB: H_ij = H_mat A_22 exp(i\Psi(f))
        for (int i=0; i<3; i++){
            for (int j=0; j<3; j++){
                trans1 = Hplus[i][j];
                trans2 = Hcross[i][j];
                H_mat[mode_i*9 + i*3 + j] = (Yfactorplus*trans1+ Yfactorcross*trans2);
                //printf("(%d, %d): %e, %e\n", i, j, Hplus[i][j], Hcross[i][j]);
            }
        }
    }

}
