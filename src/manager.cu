/*
This is the central piece of code. This file implements a class
(interface in gpuadder.hh) that takes data in on the cpu side, copies
it to the gpu, and exposes functions (increment and retreive) that let
you perform actions with the GPU

This class will get translated into python via swig
*/

#include <kernel.cu>
//#include <reduction.cu>
#include <manager.hh>
#include <assert.h>
#include <iostream>
#include "globalPhenomHM.h"
#include <complex>
#include "cuComplex.h"
#include "cublas_v2.h"
#include "interpolate.cu"


using namespace std;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


ModeContainer * gpu_create_modes(int num_modes, unsigned int *l_vals, unsigned int *m_vals, int max_length, int to_gpu, int to_interp){
        ModeContainer * cpu_mode_vals = cpu_create_modes(num_modes,  l_vals, m_vals, max_length, 1, 0);
        ModeContainer * mode_vals;

        double *amp[num_modes];
        double *phase[num_modes];

        double *amp_coeff_1[num_modes];
        double *amp_coeff_2[num_modes];
        double *amp_coeff_3[num_modes];

        double *phase_coeff_1[num_modes];
        double *phase_coeff_2[num_modes];
        double *phase_coeff_3[num_modes];


        gpuErrchk(cudaMalloc(&mode_vals, num_modes*sizeof(ModeContainer)));
        gpuErrchk(cudaMemcpy(mode_vals, cpu_mode_vals, num_modes*sizeof(ModeContainer), cudaMemcpyHostToDevice));

        for (int i=0; i<num_modes; i++){
            gpuErrchk(cudaMalloc(&amp[i], max_length*sizeof(double)));
            gpuErrchk(cudaMalloc(&phase[i], max_length*sizeof(double)));

            cudaMemcpy(&(mode_vals[i].amp), &(amp[i]), sizeof(double *), cudaMemcpyHostToDevice);
            cudaMemcpy(&(mode_vals[i].phase), &(phase[i]), sizeof(double *), cudaMemcpyHostToDevice);

            if (to_interp == 1){
                gpuErrchk(cudaMalloc(&amp_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&amp_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&amp_coeff_3[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phase_coeff_1[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phase_coeff_2[i], (max_length-1)*sizeof(double)));
                gpuErrchk(cudaMalloc(&phase_coeff_3[i], (max_length-1)*sizeof(double)));

                cudaMemcpy(&(mode_vals[i].amp_coeff_1), &(amp_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice);
                cudaMemcpy(&(mode_vals[i].amp_coeff_2), &(amp_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice);
                cudaMemcpy(&(mode_vals[i].amp_coeff_3), &(amp_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice);
                cudaMemcpy(&(mode_vals[i].phase_coeff_1), &(phase_coeff_1[i]), sizeof(double *), cudaMemcpyHostToDevice);
                cudaMemcpy(&(mode_vals[i].phase_coeff_2), &(phase_coeff_2[i]), sizeof(double *), cudaMemcpyHostToDevice);
                cudaMemcpy(&(mode_vals[i].phase_coeff_3), &(phase_coeff_3[i]), sizeof(double *), cudaMemcpyHostToDevice);
            }
        }

        return mode_vals;
}

void gpu_destroy_modes(ModeContainer * mode_vals){
    for (int i=0; i<mode_vals[0].num_modes; i++){
        gpuErrchk(cudaFree(mode_vals[i].amp));
        gpuErrchk(cudaFree(mode_vals[i].phase));
        if (mode_vals[i].to_interp == 1){
            gpuErrchk(cudaFree(mode_vals[i].amp_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].amp_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].amp_coeff_3));
            gpuErrchk(cudaFree(mode_vals[i].phase_coeff_1));
            gpuErrchk(cudaFree(mode_vals[i].phase_coeff_2));
            gpuErrchk(cudaFree(mode_vals[i].phase_coeff_3));
        }
    }
    gpuErrchk(cudaFree(mode_vals));
}


GPUPhenomHM::GPUPhenomHM (int max_length_,
    unsigned int *l_vals_,
    unsigned int *m_vals_,
    int num_modes_,
    int to_gpu_,
    int to_interp_){

    max_length = max_length_;
    l_vals = l_vals_;
    m_vals = m_vals_;
    num_modes = num_modes_;
    to_gpu = to_gpu_;
    to_interp = to_interp_;

    cudaError_t err;

    // DECLARE ALL THE  NECESSARY STRUCTS
    pHM_trans = new PhenomHMStorage;

    pAmp_trans = new IMRPhenomDAmplitudeCoefficients;

    amp_prefactors_trans = new AmpInsPrefactors;

    pDPreComp_all_trans = new PhenDAmpAndPhasePreComp[num_modes];

    q_all_trans = new HMPhasePreComp[num_modes];

  mode_vals = cpu_create_modes(num_modes, l_vals, m_vals, max_length, to_gpu, to_interp);

  if (to_gpu == 1){

      d_mode_vals = gpu_create_modes(num_modes, l_vals, m_vals, max_length, to_gpu, to_interp);

      gpuErrchk(cudaMalloc(&d_freqs, max_length*sizeof(double)));

      //gpuErrchk(cudaMalloc(&d_mode_vals, num_modes*sizeof(d_mode_vals)));
      //gpuErrchk(cudaMemcpy(d_mode_vals, mode_vals, num_modes*sizeof(d_mode_vals), cudaMemcpyHostToDevice));

      // DECLARE ALL THE  NECESSARY STRUCTS
      gpuErrchk(cudaMalloc(&d_pHM_trans, sizeof(PhenomHMStorage)));

      gpuErrchk(cudaMalloc(&d_pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients)));

      gpuErrchk(cudaMalloc(&d_amp_prefactors_trans, sizeof(AmpInsPrefactors)));

      gpuErrchk(cudaMalloc(&d_pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp)));

      gpuErrchk(cudaMalloc((void**) &d_q_all_trans, num_modes*sizeof(HMPhasePreComp)));


      double cShift[7] = {0.0,
                           PI_2 /* i shift */,
                           0.0,
                           -PI_2 /* -i shift */,
                           PI /* 1 shift */,
                           PI_2 /* -1 shift */,
                           0.0};

      gpuErrchk(cudaMalloc(&d_cShift, 7*sizeof(double)));

      gpuErrchk(cudaMemcpy(d_cShift, &cShift, 7*sizeof(double), cudaMemcpyHostToDevice));


      // for likelihood
      // --------------
      gpuErrchk(cudaMallocHost((cuDoubleComplex**) &result, sizeof(cuDoubleComplex)));

      stat = cublasCreate(&handle);
      if (stat != CUBLAS_STATUS_SUCCESS) {
              printf ("CUBLAS initialization failed\n");
              exit(0);
          }
      // ----------------
  }
  //double t0_;
  t0 = 0.0;

  //double phi0_;
  phi0 = 0.0;

  //double amp0_;
  amp0 = 0.0;
}

void GPUPhenomHM::add_interp(int max_interp_length_){
    max_interp_length = max_interp_length_;

    assert(to_interp == 1);
    if (to_gpu == 0){
        out_mode_vals = cpu_create_modes(num_modes, m_vals, l_vals, max_interp_length, to_gpu, 0);
    }
    if (to_gpu){
        d_out_mode_vals = gpu_create_modes(num_modes, m_vals, l_vals, max_interp_length, to_gpu, 0);
        gpuErrchk(cudaMalloc(&d_interp, 2*num_modes*sizeof(Interpolate)));
    }

    interp = new Interpolate[2*num_modes];
}



void GPUPhenomHM::gpu_gen_PhenomHM(double *freqs_, int f_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    assert((to_gpu == 1) || (to_gpu == 2));

    GPUPhenomHM::cpu_gen_PhenomHM(freqs_, f_length_,
        m1_, //solar masses
        m2_, //solar masses
        chi1z_,
        chi2z_,
        distance_,
        inclination_,
        phiRef_,
        deltaF_,
        f_ref_);


    // Initialize inputs
    //gpuErrchk(cudaMemcpy(d_mode_vals, mode_vals, num_modes*sizeof(ModeContainer), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_freqs, freqs, f_length*sizeof(double), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pHM_trans, pHM_trans, sizeof(PhenomHMStorage), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pAmp_trans, pAmp_trans, sizeof(IMRPhenomDAmplitudeCoefficients), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_amp_prefactors_trans, amp_prefactors_trans, sizeof(AmpInsPrefactors), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_pDPreComp_all_trans, pDPreComp_all_trans, num_modes*sizeof(PhenDAmpAndPhasePreComp), cudaMemcpyHostToDevice));

    gpuErrchk(cudaMemcpy(d_q_all_trans, q_all_trans, num_modes*sizeof(HMPhasePreComp), cudaMemcpyHostToDevice));

    double M_tot_sec = (m1+m2)*MTSUN_SI;
    /* main: evaluate model at given frequencies */
    NUM_THREADS = 256;
    num_blocks = std::ceil((f_length + NUM_THREADS -1)/NUM_THREADS);
    dim3 gridDim(num_modes, num_blocks);
    printf("blocks %d\n", num_blocks);
    kernel_calculate_all_modes<<<gridDim, NUM_THREADS>>>(d_mode_vals,
          d_pHM_trans,
          d_freqs,
          M_tot_sec,
          d_pAmp_trans,
          d_amp_prefactors_trans,
          d_pDPreComp_all_trans,
          d_q_all_trans,
          amp0,
          num_modes,
          t0,
          phi0,
          d_cShift
      );
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());

}


void GPUPhenomHM::cpu_gen_PhenomHM(double *freqs_, int f_length_,
    double m1_, //solar masses
    double m2_, //solar masses
    double chi1z_,
    double chi2z_,
    double distance_,
    double inclination_,
    double phiRef_,
    double deltaF_,
    double f_ref_){

    freqs = freqs_;
    f_length = f_length_;
    m1 = m1_; //solar masses
    m2 = m2_; //solar masses
    chi1z = chi1z_;
    chi2z = chi2z_;
    distance = distance_;
    inclination = inclination_;
    phiRef = phiRef_;
    deltaF = deltaF_;
    f_ref = f_ref_;

    for (int i=0; i<num_modes; i++){
        mode_vals[i].length = f_length;
    }

    m1_SI = m1*MSUN_SI;
    m2_SI = m2*MSUN_SI;

    /* main: evaluate model at given frequencies */
    retcode = 0;
    retcode = IMRPhenomHMCore(
        mode_vals,
        freqs,
        f_length,
        m1_SI,
        m2_SI,
        chi1z,
        chi2z,
        distance,
        inclination,
        phiRef,
        deltaF,
        f_ref,
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

void GPUPhenomHM::interp_wave(double *amp, double *phase){
    Interpolate trans;
    for (int i=0; i<num_modes; i++){
        trans.prep(freqs, &amp[i*f_length], f_length);
        trans.transferToDevice();
        interp[i] = trans;
        trans.prep(freqs,  &phase[i*f_length], f_length);
        trans.transferToDevice();
        interp[num_modes + i] = trans;
    }
    gpuErrchk(cudaMemcpy(d_interp, interp, 2*num_modes*sizeof(Interpolate), cudaMemcpyHostToDevice));


    dim3 check_dim(num_modes, num_blocks);
    int check_num_threads = 256;
    wave_interpolate<<<check_dim, check_num_threads>>>(d_interp_freqs, d_mode_vals[0].amp, d_mode_vals[0].phase, num_modes, max_interp_length, d_interp);
     cudaDeviceSynchronize();
     gpuErrchk(cudaGetLastError());


}

double GPUPhenomHM::Likelihood (){
/*
    cuComplex *trans = new cuComplex[max_length];
    for (int i; i<f_length; i++){
        trans[i] =
    }
    stat = cublasZdotc(handle, f_length,
            d_hptilde, 1,
            d_hptilde, 1,
            result);

    if (stat != CUBLAS_STATUS_SUCCESS) {
            printf ("CUBLAS initialization failed\n");
            return EXIT_FAILURE;
        }
    delete trans;
    return cuCreal(result[0]);*/
    return 0.0;
}

void GPUPhenomHM::Get_Waveform (int mode_i, double* amp_, double* phase_) {
    assert(to_gpu == 0);
    memcpy(amp_, mode_vals[mode_i].amp, f_length*sizeof(double));
    memcpy(phase_, mode_vals[mode_i].phase, f_length*sizeof(double));
}

void GPUPhenomHM::gpu_Get_Waveform (int mode_i, double* amp_, double* phase_) {
  assert(to_gpu == 1);
    gpuErrchk(cudaMemcpy(amp_, d_mode_vals[mode_i].amp, f_length*sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(phase_, d_mode_vals[mode_i].phase, f_length*sizeof(double), cudaMemcpyDeviceToHost));
}

GPUPhenomHM::~GPUPhenomHM() {
  delete pHM_trans;
  delete pAmp_trans;
  delete amp_prefactors_trans;
  delete pDPreComp_all_trans;
  delete q_all_trans;
  cpu_destroy_modes(mode_vals);

  if (to_gpu == 1){
      cudaFree(d_freqs);
      gpu_destroy_modes(d_mode_vals);
      cudaFree(d_pHM_trans);
      cudaFree(d_pAmp_trans);
      cudaFree(d_amp_prefactors_trans);
      cudaFree(d_pDPreComp_all_trans);
      cudaFree(d_q_all_trans);
      cudaFree(d_cShift);
      cudaFree(result);
      cublasDestroy(handle);
  }
  if (to_interp == 1){
      cpu_destroy_modes(out_mode_vals);
      cudaFree(d_interp);
      gpu_destroy_modes(d_out_mode_vals);
      delete interp;
  }
}
