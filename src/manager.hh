#include "globalPhenomHM.h"
#include "complex.h"
#include "assert.h"
#include "tester.hh"

/*
typedef struct tagCOMPLEX2dArray {
    double __complex__*    data;
    size_t      length;
    size_t      num_modes;
} COMPLEX2dArray;

COMPLEX2dArray *CreateCOMPLEX2dArray(
    size_t length,
    size_t num_modes
){
    COMPLEX2dArray* tmp = (COMPLEX2dArray*)malloc(sizeof(COMPLEX2dArray));
    tmp->length = length;
    tmp->num_modes = num_modes;
    size_t size = sizeof(double complex)*num_modes*length;
    tmp->data = (double complex*)malloc(size);

    assert(tmp->data != NULL);
        //ERROR(PD_ENOMEM, "Failed to allocated data array.");

    memset(tmp->data, 0, sizeof(size));
    //memset(tmp->data, 0, sizeof(double complex)*(num_modes * length));
    return tmp;
}

void DestroyCOMPLEX2dArray(COMPLEX2dArray *tmp) {
    free(tmp->data);
    free(tmp);
}
*/

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
  double complex *factorp_trans;
  double complex *factorc_trans;
  double t0;
  double phi0;
  double amp0;
  //COMPLEX2dArray *hptilde;
  //COMPLEX2dArray *hctilde;

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
      double *freqs, int f_length_,
      double m1, //solar masses
      double m2, //solar masses
      double chi1z,
      double chi2z,
      double distance,
      double inclination,
      double phiRef,
      double deltaF,
      double f_ref,
      unsigned int *l_vals,
      unsigned int *m_vals,
      int num_modes,
      int to_gpu); // constructor (copies to GPU)

  ~GPUPhenomHM(); // destructor

  void increment(); // does operation inplace on the GPU

  void retreive(); //gets results back from GPU, putting them in the memory that was passed in
  // the constructor

  //gets results back from the gpu, putting them in the supplied memory location
  void retreive_to (int* INPLACE_ARRAY1, int DIM1);


};
