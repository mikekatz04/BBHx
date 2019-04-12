#include "globalPhenomHM.h"
#include "complex.h"

#include "assert.h"
#include "tester.hh"
#include "PhenomHM.h"

class GPUPhenomHM {
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
  int to_gpu;
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

  // pointer to the GPU memory where the array is stored
  int* array_device;
  // pointer to the CPU memory where the array is stored
  int* array_host;
  // length of the array (number of elements)
  int length;

  StructTest *x;
  StructTest *d_x;

public:
  /* By using the swig default names INPLACE_ARRAY1, DIM1 in the header
     file (these aren't the names in the implementation file), we're giving
     swig the info it needs to cast to and from numpy arrays.

     If instead the constructor line said
       GPUAdder(int* myarray, int length);

     We would need a line like this in the swig.i file
       %apply (int* ARGOUT_ARRAY1, int DIM1) {(int* myarray, int length)}
   */

  GPUPhenomHM(int* INPLACE_ARRAY1, int DIM1,
      double *freqs_, int f_length_,
      unsigned int *l_vals_,
      unsigned int *m_vals_,
      int num_modes_,
      int to_gpu_); // constructor (copies to GPU)

  ~GPUPhenomHM(); // destructor

  void increment(); // does operation inplace on the GPU
  void cpu_gen_PhenomHM(
        double m1_, //solar masses
        double m2_, //solar masses
        double chi1z_,
        double chi2z_,
        double distance_,
        double inclination_,
        double phiRef_,
        double deltaF_,
        double f_ref_);
  void retreive(); //gets results back from GPU, putting them in the memory that was passed in
  // the constructor
  void retreive_to(int* INPLACE_ARRAY1, int DIM1); //gets results back from GPU, putting them in the memory that was passed in
  // the constructor

  //gets results back from the gpu, putting them in the supplied memory location
  void Get_Waveform (std::complex<double>* hptilde_, std::complex<double>* hctilde_);


};
