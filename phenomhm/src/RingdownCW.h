#ifndef _RINGDOWN_CW_H
#define _RINGDOWN_CW_H
/* ************************************************************  */
/*  This code was edited by Michael Katz. It is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited
 *  the code for his purposes and removed dependencies on the LAL libraries. The code has been confirmed to match the LAL version.
 *  This code is distrbuted under the same GNU license it originally came with.
 *  The comments in the code have been left generally the same. A few comments
 *  have been made for the newer functions added.

 
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

//#if defined(__cplusplus)
//extern "C" {
//#elif 0
//} /* so that editors will match preceding brace */
//#endif

/* Include the desired Libs */
#include <stdbool.h>
#include <math.h>
#include <complex>
#include "globalPhenomHM.h"

/*
* Basic NOTE(s): Prototypes for LALSimRingdownCW.c
*/

/*
* Domain mapping for dimnesionless BH spin
*/
double SimRingdownCW_KAPPA(double jf, int l, int m);

/*
* Dimensionless QNM Frequencies: Note that name encodes date of writing
*/
cmplx SimRingdownCW_CW07102016(double kappa, /* Domain mapping for  remnant BH's spin (Dimensionless) */
                                        int l,        /* Polar eigenvalue */
                                        int input_m,  /* Azimuthal eigenvalue*/
                                        int n);       /* Overtone Number*/

/* ************************************************************  */


//#if 0
//{ /* so that editors will match succeeding brace */
//#elif defined(__cplusplus)
//}
//#endif

#endif /* _LALSIM_RINGDOWN_CW_H */
