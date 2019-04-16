#include "globalPhenomHM.h"
#include <complex>
#include "assert.h"
#include "tester.hh"
#include "PhenomHM.h"

class PhenomHM {
  // pointer to the GPU memory where the array is stored
  double *freqs;
  int f_length;
  double m1; //solar masses
  double m2; //solar masses
  double chi1z;
  double chi2z;
  double distance;
  double inclination;
  double phiRef;
  double deltaF;
  double f_ref;
  unsigned int *l_vals;
  unsigned int *m_vals;
  int num_modes;
  PhenomHMStorage *pHM_trans;
  IMRPhenomDAmplitudeCoefficients *pAmp_trans;
  AmpInsPrefactors *amp_prefactors_trans;
  PhenDAmpAndPhasePreComp *pDPreComp_all_trans;
  HMPhasePreComp *q_all_trans;
  std::complex<double> *factorp_trans;
  std::complex<double> *factorc_trans;
  double t0;
  double phi0;
  double amp0;
  int retcode;
  double m1_SI;
  double m2_SI;
  std::complex<double> *hptilde;
  std::complex<double> *hctilde;
  double *freqs_geom_trans;


public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  PhenomHM(double *freqs_, int f_length_,
      unsigned int *l_vals_,
      unsigned int *m_vals_,
      int num_modes_); // constructor (copies to )

  ~PhenomHM(); // destructor

  void gen_PhenomHM(
        double m1_, //solar masses
        double m2_, //solar masses
        double chi1z_,
        double chi2z_,
        double distance_,
        double inclination_,
        double phiRef_,
        double deltaF_,
        double f_ref_);

  double Likelihood ();
  //gets results back from the gpu, putting them in the supplied memory location
  void Get_Waveform (std::complex<double>* hptilde_, std::complex<double>* hctilde_);

};
