/*
* Copyright (C) 2016 Lionel London
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

/* .................... */
/* HEADER SECTION       */
/* .................... */

#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <complex>
#include <stdlib.h>

#include "globalPhenomHM.h"
#include "RingdownCW.h"

/*
* Based on the paper by London and Fauchon-Jones: https://arxiv.org/abs/1810.03550
* Basic NOTE(s):
*   - This file contains a function, CW07102016, which outputs complex valued, UNITLESS, QNM frequencies (i.e. Mw) for various QNMs
*   - Usage: cw = CW07102016( kappa, l, m, n ); where cw = Mw + 1i*M/tau; NOTE that kappa is a function of final spin, l and m
*   - See definition of KAPPA below.
*/

/*
* -------------------------------------------------------------------------------- *
* Low level models: QNM Frequencies
* -------------------------------------------------------------------------------- *
*/

/*
* Domain mapping for dimnesionless BH spin
*/
double SimRingdownCW_KAPPA(double jf, int l, int m)
{
    /* */
    /* if ( jf > 1.0 ) ERROR(XLAL_EDOM, "Spin (dimensionless Kerr parameter) must not be greater than 1.0\n"); */
    /**/
    double alpha = log(2.0 - jf) / log(3);
    double beta = 1.0 / (2.0 + l - abs(m));
    return pow(alpha, beta);
}

/*
* Dimensionless QNM Frequencies: Note that name encodes date of writing
*/
/*TODO: Make the function arg comments compatible with doxygen*/
cmplx SimRingdownCW_CW07102016(double kappa, /* Domain mapping for  remnant BH's spin (Dimensionless) */
                                        int l,        /* Polar eigenvalue */
                                        int input_m,  /* Azimuthal eigenvalue*/
                                        int n)
{ /* Overtone Number*/

    /* Predefine powers to increase efficiency*/
    double kappa2 = kappa * kappa;
    double kappa3 = kappa2 * kappa;
    double kappa4 = kappa3 * kappa;

    /* NOTE that |m| will be used to determine the fit to use, and if input_m < 0, then a conjugate will be taken*/
    int m = abs(input_m);

    /**/
    cmplx j = cmplx(0.0, 1.0);

    /* Initialize the answer*/
    cmplx ans;

    /* Use If-Else ladder to determine which mode function to evaluate*/
    if (2 == l && 2 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (2,2,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.0 + kappa * (1.557847 * std::exp(2.903124 * j) +
                             1.95097051 * std::exp(5.920970 * j) * kappa +
                             2.09971716 * std::exp(2.760585 * j) * kappa2 +
                             1.41094660 * std::exp(5.914340 * j) * kappa3 +
                             0.41063923 * std::exp(2.795235 * j) * kappa4);
    }
    else if (2 == l && 2 == m && 1 == n)
    {

        /* Fit for (l,m,n) == (2,2,1). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.0 + kappa * (1.870939 * std::exp(2.511247 * j) +
                             2.71924916 * std::exp(5.424999 * j) * kappa +
                             3.05648030 * std::exp(2.285698 * j) * kappa2 +
                             2.05309677 * std::exp(5.486202 * j) * kappa3 +
                             0.59549897 * std::exp(2.422525 * j) * kappa4);
    }
    else if (3 == l && 2 == m && 0 == n)
    {

        /* Define extra powers as needed*/
        double kappa5 = kappa4 * kappa;
        double kappa6 = kappa5 * kappa;

        /* Fit for (l,m,n) == (3,2,0). This is NOT a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.022464 * std::exp(0.004870 * j) +
              0.24731213 * std::exp(0.665292 * j) * kappa +
              1.70468239 * std::exp(3.138283 * j) * kappa2 +
              0.94604882 * std::exp(0.163247 * j) * kappa3 +
              1.53189884 * std::exp(5.703573 * j) * kappa4 +
              2.28052668 * std::exp(2.685231 * j) * kappa5 +
              0.92150314 * std::exp(5.841704 * j) * kappa6;
    }
    else if (4 == l && 4 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (4,4,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 2.0 + kappa * (2.658908 * std::exp(3.002787 * j) +
                             2.97825567 * std::exp(6.050955 * j) * kappa +
                             3.21842350 * std::exp(2.877514 * j) * kappa2 +
                             2.12764967 * std::exp(5.989669 * j) * kappa3 +
                             0.60338186 * std::exp(2.830031 * j) * kappa4);
    }
    else if (2 == l && 1 == m && 0 == n)
    {

        /* Define extra powers as needed*/
        double kappa5 = kappa4 * kappa;
        double kappa6 = kappa5 * kappa;

        /* Fit for (l,m,n) == (2,1,0). This is NOT a zero-damped mode in the extremal Kerr limit.*/
        ans = 0.589113 * std::exp(0.043525 * j) +
              0.18896353 * std::exp(2.289868 * j) * kappa +
              1.15012965 * std::exp(5.810057 * j) * kappa2 +
              6.04585476 * std::exp(2.741967 * j) * kappa3 +
              11.12627777 * std::exp(5.844130 * j) * kappa4 +
              9.34711461 * std::exp(2.669372 * j) * kappa5 +
              3.03838318 * std::exp(5.791518 * j) * kappa6;
    }
    else if (3 == l && 3 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (3,3,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (2.095657 * std::exp(2.964973 * j) +
                             2.46964352 * std::exp(5.996734 * j) * kappa +
                             2.66552551 * std::exp(2.817591 * j) * kappa2 +
                             1.75836443 * std::exp(5.932693 * j) * kappa3 +
                             0.49905688 * std::exp(2.781658 * j) * kappa4);
    }
    else if (3 == l && 3 == m && 1 == n)
    {

        /* Fit for (l,m,n) == (3,3,1). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (2.339070 * std::exp(2.649692 * j) +
                             3.13988786 * std::exp(5.552467 * j) * kappa +
                             3.59156756 * std::exp(2.347192 * j) * kappa2 +
                             2.44895997 * std::exp(5.443504 * j) * kappa3 +
                             0.70040804 * std::exp(2.283046 * j) * kappa4);
    }
    else if (4 == l && 3 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (4,3,0). This is a zero-damped mode in the extremal Kerr limit.*/
        ans = 1.5 + kappa * (0.205046 * std::exp(0.595328 * j) +
                             3.10333396 * std::exp(3.016200 * j) * kappa +
                             4.23612166 * std::exp(6.038842 * j) * kappa2 +
                             3.02890198 * std::exp(2.826239 * j) * kappa3 +
                             0.90843949 * std::exp(5.915164 * j) * kappa4);
    }
    else if (5 == l && 5 == m && 0 == n)
    {

        /* Fit for (l,m,n) == (5,5,0). This is a zero-damped mode in the extremal Kerr limit. */
        ans = 2.5 + kappa * (3.240455 * std::exp(3.027869 * j) +
                             3.49056455 * std::exp(6.088814 * j) * kappa +
                             3.74704093 * std::exp(2.921153 * j) * kappa2 +
                             2.47252790 * std::exp(6.036510 * j) * kappa3 +
                             0.69936568 * std::exp(2.876564 * j) * kappa4);
    }
    else
    {

        /**/
        ans = 0.0;

    } /* END of IF-ELSE Train for QNM cases */

    /* If m<0, then take the *Negative* conjugate */
    if (input_m < 0)
    {
        /**/
        ans = -conj(ans);
    }

    return ans;

} /* END of CW07102016 */
