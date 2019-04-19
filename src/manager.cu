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
        h_indices = new int[max_interp_length];
        cudaMalloc(&d_indices, max_interp_length*sizeof(int));
        d_out_mode_vals = gpu_create_modes(num_modes, m_vals, l_vals, max_interp_length, to_gpu, 0);
    }
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

__global__ void fill_B(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= f_length) return;
    if (mode_i >= num_modes) return;
    if (i == f_length - 1){
        B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i] - mode_vals[mode_i].amp[i-1]);
        B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i] - mode_vals[mode_i].phase[i-1]);
    } else if (i == 0){
        B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[1] - mode_vals[mode_i].amp[0]);
        B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[1] - mode_vals[mode_i].phase[0]);
    } else{
        B[mode_i*f_length + i] = 3.0* (mode_vals[mode_i].amp[i+1] - mode_vals[mode_i].amp[i-1]);
        B[(num_modes*f_length) + mode_i*f_length + i] = 3.0* (mode_vals[mode_i].phase[i+1] - mode_vals[mode_i].phase[i-1]);
    }


}

__global__ void set_spline_constants(ModeContainer *mode_vals, double *B, int f_length, int num_modes){
    int mode_i = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= f_length-1) return;
    if (mode_i >= num_modes) return;
    double D_i, D_ip1, y_i, y_ip1;

    D_i = B[mode_i*f_length + i];
    D_ip1 = B[mode_i*f_length + i + 1];
    y_i = mode_vals[mode_i].amp[i];
    y_ip1 = mode_vals[mode_i].amp[i+1];
    mode_vals[mode_i].amp_coeff_1[i] = D_i;
    mode_vals[mode_i].amp_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
    mode_vals[mode_i].amp_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;

    D_i = B[(num_modes*f_length) + mode_i*f_length + i];
    D_ip1 = B[(num_modes*f_length) + mode_i*f_length + i + 1];
    y_i = mode_vals[mode_i].phase[i];
    y_ip1 = mode_vals[mode_i].phase[i+1];
    mode_vals[mode_i].phase_coeff_1[i] = D_i;
    mode_vals[mode_i].phase_coeff_2[i] = 3.0 * (y_ip1 - y_i) - 2.0*D_i - D_ip1;
    mode_vals[mode_i].phase_coeff_3[i] = 2.0 * (y_i - y_ip1) + D_i + D_ip1;
}

__global__ void interpolate(ModeContainer* old_mode_vals, ModeContainer* new_mode_vals, int ind_min, int ind_max, int num_modes, double f_min, double df, int old_ind_below, double *old_freqs){
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.x;
    if (i + ind_min > ind_max) return;
    if (mode_i >= num_modes) return;
    int new_index = i + ind_min;
    double f = f_min + df * new_index;
    double x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
    double x2 = x*x;
    double x3 = x*x2;
    double coeff_0, coeff_1, coeff_2, coeff_3;
    // interp amplitude
    coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
    coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
    coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
    coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

    new_mode_vals[mode_i].amp[new_index] = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

    // interp phase
    coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
    coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
    coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
    coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

    new_mode_vals[mode_i].phase[new_index] = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
}

__global__ void interpolate2(ModeContainer* old_mode_vals, ModeContainer* new_mode_vals, int ind_min, int ind_max, int num_modes, double f_min, double df, int *old_inds, double *old_freqs){
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    int mode_i = blockIdx.x;
    if (i + ind_min > ind_max) return;
    if (mode_i >= num_modes) return;
    int new_index = i + ind_min;
    int old_ind_below = old_inds[i];
    double f = f_min + df * new_index;
    double x = (f - old_freqs[old_ind_below])/(old_freqs[old_ind_below+1] - old_freqs[old_ind_below]);
    double x2 = x*x;
    double x3 = x*x2;
    double coeff_0, coeff_1, coeff_2, coeff_3;
    // interp amplitude
    coeff_0 = old_mode_vals[mode_i].amp[old_ind_below];
    coeff_1 = old_mode_vals[mode_i].amp_coeff_1[old_ind_below];
    coeff_2 = old_mode_vals[mode_i].amp_coeff_2[old_ind_below];
    coeff_3 = old_mode_vals[mode_i].amp_coeff_3[old_ind_below];

    new_mode_vals[mode_i].amp[new_index] = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);

    // interp phase
    coeff_0 = old_mode_vals[mode_i].phase[old_ind_below];
    coeff_1 = old_mode_vals[mode_i].phase_coeff_1[old_ind_below];
    coeff_2 = old_mode_vals[mode_i].phase_coeff_2[old_ind_below];
    coeff_3 = old_mode_vals[mode_i].phase_coeff_3[old_ind_below];

    new_mode_vals[mode_i].phase[new_index] = coeff_0 + (coeff_1*x) + (coeff_2*x2) + (coeff_3*x3);
}

__global__ void read_out_kernel2(ModeContainer *mode_vals, double *coef0, double *coef1, double *coef2, double *coef3, int mode_i, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    coef0[i] = mode_vals[mode_i].amp[i];
    coef1[i] = mode_vals[mode_i].amp_coeff_1[i];
    coef2[i] = mode_vals[mode_i].amp_coeff_2[i];
    coef3[i] = mode_vals[mode_i].amp_coeff_3[i];
    //phase[i] = mode_vals[mode_i].phase[i];
}

void GPUPhenomHM::interp_wave(double f_min, double df, int length_new){
    //printf("%e, %e\n", f_min, df);
    double *d_B;//, *h_B, *h_B1;
    //h_B = new double[2*f_length*num_modes];
    //h_B1 = new double[2*f_length*num_modes];*/
    gpuErrchk(cudaMallocManaged(&d_B, 2*f_length*num_modes*sizeof(double)));

    dim3 check_dim(num_modes, num_blocks);
    int check_num_threads = 256;
    fill_B<<<check_dim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    //cudaMemcpy(h_B1, d_B, 2*f_length*num_modes*sizeof(double), cudaMemcpyDeviceToHost);
    //for (int i=0; i<2*f_length*num_modes; i++) printf("%e\n", h_B1[i]);
    //printf("before\n");
    interp.prep(d_B, f_length, 2*num_modes, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    //double * d_B_amp_only, *h_B_amp_only,* h_B_amp_only_transfer;
    /*
    h_B_amp_only = new double[f_length];
    h_B_amp_only_transfer = new double[f_length];
    gpuErrchk(cudaMalloc(&d_B_amp_only, f_length*sizeof(double)));

    gpuErrchk(cudaMemcpy(d_B_amp_only, d_B, f_length*sizeof(double), cudaMemcpyDeviceToDevice));
    gpuErrchk(cudaMemcpy(h_B_amp_only, d_B, f_length*sizeof(double), cudaMemcpyDeviceToHost));
    interp.prep(d_B_amp_only, f_length, 1, 1);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaMemcpy(h_B_amp_only_transfer, d_B_amp_only, f_length*sizeof(double), cudaMemcpyDeviceToHost));

    interp.prep(h_B_amp_only, f_length, 1, 0);
    //cudaMemcpy(h_B, d_B, 2*f_length*num_modes*sizeof(double), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    printf("check\n");
    for (int i=0; i<f_length; i++){
        printf("%d %e %e\n", i, h_B_amp_only[i], h_B_amp_only_transfer[i]);
    }*/
    //printf("before\n");
    set_spline_constants<<<check_dim, NUM_THREADS>>>(d_mode_vals, d_B, f_length, num_modes);
    //printf("after fillB\n");
    /*cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    int length_check = 10;
    double *d_coef0, *d_coef1, *d_coef2, *d_coef3; //, *h_coef0,  *h_coef1, *h_coef2, *h_coef3;
    num_blocks = (int)((length_check + NUM_THREADS -1 )/NUM_THREADS);

    cudaMallocManaged(&d_coef0, length_check*sizeof(double));
    cudaMallocManaged(&d_coef1, length_check*sizeof(double));
    cudaMallocManaged(&d_coef2, length_check*sizeof(double));
    cudaMallocManaged(&d_coef3, length_check*sizeof(double));
    read_out_kernel2<<<num_blocks, NUM_THREADS>>>(d_mode_vals, d_coef0, d_coef1, d_coef2, d_coef3, 0, length_check);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    //for (i=0; i<length_check; i++) printf("%e, %e, %e, %e, %e\n", freqs[i], d_coef0[i], d_coef1[i], d_coef2[i], d_coef3[i]);
    cudaFree(d_coef0);
    cudaFree(d_coef1);
    cudaFree(d_coef2);
    cudaFree(d_coef3);*/
    /*int i = 0;
    double f = f_min;
    while (f < freqs[0]){
        i++;
        f = f_min + df*i;
    }

    int num_evals, threads;
    int old_index = 0;
    int new_start_index = 0;
    int new_end_index = 1;
    for (i; i<length_new; i++){
        f = f_min + df*i;
        if ((f > freqs[old_index+1]) || (i == length_new - 1)){
            if (f > freqs[old_index+1]) new_end_index = i-1;
            else new_end_index = i;
            num_evals = new_end_index - new_start_index + 1;

            if (num_evals >= 128) threads = 256;
            else if (num_evals >= 64) threads = 128;
            else if (num_evals >= 32) threads = 64;
            else threads = 32; //  warps are in factors of 32 so set this to minimum
            num_blocks = (int) ((num_evals + threads - 1) / threads);
            //printf("blocks: %d, threads %d, evals: %d, start: %d, end: %d\n", num_blocks, threads, num_evals, new_start_index, new_end_index);
            dim3 gridDim(num_modes, num_blocks);
            cudaDeviceSynchronize();
            gpuErrchk(cudaGetLastError());
            interpolate<<<gridDim, threads>>>(d_mode_vals,
                d_out_mode_vals,
                new_start_index,
                new_end_index,
                num_modes,
                f_min,
                df,
                old_index,
                d_freqs);
            new_start_index = i;
            old_index++;
            if (old_index >= f_length) break;
        }
    }
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());*/

    int i = 0;
    double f = f_min;
    while (f < freqs[0]){
        i++;
        f = f_min + df*i;
    }
    int new_start_index = i;
    int new_end_index;
    int ended = 0;
    int index = 0;
    for (i; i<length_new; i++){
        if (f > freqs[f_length-1]){
            new_end_index = i-1;
            ended = 1;
            break;
        }
        f = f_min + df*i;
        if (f < freqs[index + 1]){
            h_indices[i] = index;
        } else{
            index++;
            h_indices[i] = index;
        }
    }
    if (ended == 0) new_end_index = i;
    int num_evals = new_end_index - new_start_index + 1;
    num_blocks = (int) ((num_evals + NUM_THREADS - 1) / NUM_THREADS);
    dim3 gridDim(num_modes, num_blocks);
    gpuErrchk(cudaMemcpy(&d_indices[new_start_index], &h_indices[new_start_index], length_new*sizeof(int), cudaMemcpyHostToDevice));
    interpolate2<<<gridDim, NUM_THREADS>>>(d_mode_vals,
        d_out_mode_vals,
        new_start_index,
        new_end_index,
        num_modes,
        f_min,
        df,
        d_indices,
        d_freqs);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    /*int num_blocks_interp = ceil(length_new + NUM_THREADS -1)/NUM_THREADS;
    dim3 interp_dim(num_modes, num_blocks_interp, f_length);
    int *h_ind_out, *d_ind_out;
    h_ind_out = new int[length_new*sizeof(int)];
    cudaMalloc(&d_ind_out, length_new*sizeof(int));
    printf("%e, %e\n", f_min, df);
    interpolate2<<<interp_dim, NUM_THREADS>>>(d_freqs, f_min, df, f_length, length_new, num_modes, d_ind_out);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaMemcpy(h_ind_out, d_ind_out, length_new*sizeof(int), cudaMemcpyDeviceToHost);
    //printf("after\n");
    //wave_interpolate<<<check_dim, check_num_threads>>>(d_interp_freqs, d_mode_vals[0].amp, d_mode_vals[0].phase, num_modes, max_interp_length, d_interp);
    double f;

    printf("%e, %e\n", f_min, df);
    for (int i=0; i<length_new; i++){
        f = f_min + df*i;
        if (i % 100000 == 0) printf("%d, %e, %d\n", i, f, h_ind_out[i]);
    }*/



    //delete h_B;
    //delete h_B1;
    //delete h_ind_out;
    cudaFree(d_B);
    //cudaFree(d_ind_out);
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

__global__ void read_out_kernel(ModeContainer *mode_vals, double *amp, double *phase, int mode_i, int length){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;
    amp[i] = mode_vals[mode_i].amp[i];
    phase[i] = mode_vals[mode_i].phase[i];
}

void GPUPhenomHM::gpu_Get_Waveform (int mode_i, double* amp_, double* phase_) {
  assert(to_gpu == 1);
  double *amp;
  double *phase;
  int num_blocks = (int)((max_interp_length + NUM_THREADS - 1)/NUM_THREADS);
  //gpuErrchk(cudaMalloc(&amp, f_length*sizeof(double)));
  //gpuErrchk(cudaMalloc(&phase, f_length*sizeof(double)));
  //read_out_kernel<<<num_blocks,NUM_THREADS>>>(d_mode_vals, amp, phase, mode_i, f_length);
  gpuErrchk(cudaMalloc(&amp, max_interp_length*sizeof(double)));
  gpuErrchk(cudaMalloc(&phase, max_interp_length*sizeof(double)));
  read_out_kernel<<<num_blocks,NUM_THREADS>>>(d_out_mode_vals, amp, phase, mode_i, max_interp_length);

  cudaDeviceSynchronize();
  gpuErrchk(cudaGetLastError());
  /*double *amp;
  double *phase;
  gpuErrchk(cudaMalloc(&amp, f_length*sizeof(double)));
  gpuErrchk(cudaMalloc(&phase, f_length*sizeof(double)));

  cudaMemcpy(&(amp), &(mode_vals[mode_i].amp),sizeof(double *), cudaMemcpyDeviceToHost);
  cudaMemcpy(&(phase), &(mode_vals[mode_i].phase), sizeof(double *), cudaMemcpyDeviceToHost);

    gpuErrchk(cudaMemcpy(amp_, amp, f_length*sizeof(double), cudaMemcpyDeviceToHost));

    gpuErrchk(cudaMemcpy(phase_, phase, f_length*sizeof(double), cudaMemcpyDeviceToHost));
    */
    //printf("max_interp_length: %d \n", max_interp_length);
    gpuErrchk(cudaMemcpy(amp_, amp, max_interp_length*sizeof(double), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(phase_, phase, max_interp_length*sizeof(double), cudaMemcpyDeviceToHost));
    cudaFree(amp);
    cudaFree(phase);

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
      delete h_indices;
      cudaFree(d_indices);
      cpu_destroy_modes(out_mode_vals);
      gpu_destroy_modes(d_out_mode_vals);
      //delete interp;
  }
}
