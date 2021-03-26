#include "EOB.hh"
#include "global.h"
#include "constants.h"

#define NUM_THREADS_EOB 256
#define MAX_MODES 22

CUDA_KERNEL
void compute_hlms(cmplx* hlms, double* r_arr, double* phi_arr, double* pr_arr, double* L_arr,
                  double* m1_arr, double* m2_arr, double* chi1_arr, double* chi2_arr,
                  int* num_steps, int num_steps_max, int* ell_arr_in, int* mm_arr_in, int num_modes, int num_bin_all)
{

    CUDA_SHARED int ell_arr[MAX_MODES];
    CUDA_SHARED int mm_arr[MAX_MODES];

    cmplx I(0.0, 1.0);

    int start, increment;

    #ifdef __CUDACC__
    start = threadIdx.x;
    increment = blockDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int i = start; i < num_modes; i += increment)
    {
        ell_arr[i] = ell_arr_in[i];
        mm_arr[i] = mm_arr_in[i];
    }
    CUDA_SYNC_THREADS;

    #ifdef __CUDACC__
    start = blockIdx.x;
    increment = gridDim.x;
    #else
    start = 0;
    increment = 1;
    #pragma omp parallel for
    #endif
    for (int bin_i = start; bin_i < num_bin_all; bin_i += increment)
    {
        int num_steps_here = num_steps[bin_i];

        double m_1 = m1_arr[bin_i];
        double m_2 = m2_arr[bin_i];
        double chi_1 = chi1_arr[bin_i];
        double chi_2 = chi2_arr[bin_i];

        double M = m_1 + m_2;
        double mu = m_1*m_2/(m_1+m_2);
        double nu = mu/M;
        double X1 = m_1/M;
        double X2 = m_2/M;
        double Chi1 = chi_1;
        double Chi2 = chi_2;
        double Nu = nu;
        double Delta = (m_1-m_2)/M;

        int start2, increment2;
        #ifdef __CUDACC__
        start2 = threadIdx.x;
        increment2 = blockDim.x;
        #else
        start2 = 0;
        increment2 = 1;
        #pragma omp parallel for
        #endif
        for (int i = start2; i < num_steps_here; i += increment2)
        {
            double r = r_arr[bin_i * num_steps_max + i];
            double phi = r_arr[bin_i * num_steps_max + i];
            double pr = r_arr[bin_i * num_steps_max + i];
            double L = r_arr[bin_i * num_steps_max + i];

            cmplx HCirc_temp(0.0, 0.0);
            for (int mode_i = 0; mode_i < num_modes; mode_i += 1)
            {
                int ell = ell_arr[mode_i];
                int mm = mm_arr[mode_i];

                if ((ell == 2) && (mm == 0))
                    HCirc_temp = cmplx(0.0, 0.0);

                else if ((ell == 2) && (mm == 1))
                    HCirc_temp = -0.25*I*Chi1*(Delta + 1.)/pow(r, 2) + 0.0119047619047619*I*Chi1*(Delta*(26.*Nu - 21.) + 247.*Nu - 21)/pow(r, 3) - 0.25*I*Chi2*(Delta - 1)/pow(r, 2) + 0.0119047619047619*I*Chi2*(Delta*(26*Nu - 21) - 247*Nu + 21)/pow(r, 3) + 0.333333333333333*I*Delta/pow(r, 3./2.) + 0.202380952380952*I*Delta*(2*Nu - 1)/pow(r, 5./2.) - 0.666666666666667*sqrt(2)*I*PI/pow(r, 3);

                else if ((ell == 2) && (mm == 2))
                    HCirc_temp = 0.25*pow(Chi1, 2)*(3.0*Delta - 6.0*Nu + 3.0)/pow(r, 3) + 3.0*Chi1*Chi2*Nu/pow(r, 3) + 0.166666666666667*Chi1*(-6*Delta + 5*Nu - 6)/pow(r, 5./2.) - 0.25*pow(Chi2, 2)*(3.0*Delta + 6.0*Nu - 3.0)/pow(r, 3) + 0.166666666666667*Chi2*(6*Delta + 5*Nu - 6)/pow(r, 5./2.) + 1/r + (69.0*Nu - 107.0)/(42*pow(r, 2)) + 0.000661375661375661*(3703*pow(Nu, 2) - 11941*Nu - 2173)/pow(r, 3) + 2.0*PI/pow(r, 5./2.);

                else if ((ell == 3) && (mm == 0))
                    HCirc_temp = cmplx(0.0, 0.0);

                else if ((ell == 3) && (mm == 1))
                    HCirc_temp = 0.00148809523809524*sqrt(14)*I*Chi1*(2*Delta*(7*Nu - 3) + 19*Nu - 6)/pow(r, 3) + 0.00148809523809524*sqrt(14)*I*Chi2*(2*Delta*(7*Nu - 3) - 19*Nu + 6)/pow(r, 3) + 0.00595238095238095*sqrt(14)*I*Delta/pow(r, 3./2.) - 0.000992063492063492*sqrt(14)*I*Delta*(Nu + 16)/pow(r, 5./2.) - 0.00595238095238095*sqrt(14)*I*PI/pow(r, 3); // 3,1]

                else if ((ell == 3) && (mm == 2))
                    HCirc_temp = 0.0952380952380952*sqrt(35)*Chi1*Nu/pow(r, 5./2.) + 0.0952380952380952*sqrt(35)*Chi2*Nu/pow(r, 5./2.) + 0.0476190476190476*sqrt(35)*(1 - 3*Nu)/pow(r, 2) - sqrt(35)*(545.0*pow(Nu, 2) - 785.0*Nu + 193.0)/(1890*pow(r, 3)); // [3,2]

                else if ((ell == 3) && (mm == 3))
                    HCirc_temp = -0.0401785714285714*sqrt(210)*I*Chi1*(2*Delta*(Nu - 1) + 9*Nu - 2)/pow(r, 3) - 0.0401785714285714*sqrt(210)*I*Chi2*(2*Delta*(Nu - 1) - 9*Nu + 2)/pow(r, 3) - 0.0535714285714286*sqrt(210)*I*Delta/pow(r, 3./2.) - 0.0267857142857143*sqrt(210)*I*Delta*(5*Nu - 8)/pow(r, 5./2.) + 0.160714285714286*sqrt(210)*I*PI/pow(r, 3);

                else if ((ell == 4) && (mm == 0))
                    HCirc_temp = cmplx(0.0, 0.0);

                else if ((ell == 4) && (mm == 1))
                    HCirc_temp = 0.00148809523809524*sqrt(10)*I*Chi1*Nu*(Delta - 1)/pow(r, 3) + 0.00148809523809524*sqrt(10)*I*Chi2*Nu*(Delta + 1)/pow(r, 3) - 0.00119047619047619*sqrt(10)*I*Delta*(2*Nu - 1)/pow(r, 5./2.);

                else if ((ell == 4) && (mm == 2))
                    HCirc_temp = 0.0158730158730159*sqrt(5)*(1 - 3*Nu)/pow(r, 2) - sqrt(5)*(315.0*pow(Nu, 2) - 1415.0*Nu + 437.0)/(6930*pow(r, 3));

                else if ((ell == 4) && (mm == 3))
                    HCirc_temp = -0.0401785714285714*sqrt(70)*I*Chi1*Nu*(Delta - 1)/pow(r, 3) - 0.0401785714285714*sqrt(70)*I*Chi2*Nu*(Delta + 1)/pow(r, 3) + 0.0321428571428571*sqrt(70)*I*Delta*(2*Nu - 1)/pow(r, 5./2.);

                else if ((ell == 4) && (mm == 4))
                    HCirc_temp = 0.126984126984127*sqrt(35)*(3*Nu - 1)/pow(r, 2) + sqrt(35)*(4380.0*pow(Nu, 2) - 8780.0*Nu + 2372.0)/(3465*pow(r, 3));

                else if ((ell == 5) && (mm == 0))
                    HCirc_temp = cmplx(0.0, 0.0);

                else if ((ell == 5) && (mm == 1))
                    HCirc_temp = -9.01875901875902e-6*sqrt(385)*I*Delta*(2*Nu - 1)/pow(r, 5./2.);

                else if ((ell == 5) && (mm == 2))
                    HCirc_temp = sqrt(55)*(10.0*pow(Nu, 2) - 10.0*Nu + 2.0)/(1485*pow(r, 3));

                else if ((ell == 5) && (mm == 3))
                    HCirc_temp = 0.00255681818181818*sqrt(330)*I*Delta*(2*Nu - 1)/pow(r, 5./2.);

                else if ((ell == 5) && (mm == 4))
                    HCirc_temp = sqrt(165)*(-160.0*pow(Nu, 2) + 160.0*Nu - 32.0)/(1485*pow(r, 3));

                else if ((ell == 5) && (mm == 5))
                    HCirc_temp = -0.0986426767676768*sqrt(66)*I*Delta*(2*Nu - 1)/pow(r, 5./2.);

                else if ((ell == 6) && (mm == 0))
                    HCirc_temp = cmplx(0.0, 0.0);

                else if ((ell == 6) && (mm == 2))
                    HCirc_temp = sqrt(65)*(10.0*pow(Nu, 2) - 10.0*Nu + 2.0)/(19305*pow(r, 3));

                else if ((ell == 6) && (mm == 4))
                    HCirc_temp = -0.00663040663040663*sqrt(78)*(5*pow(Nu, 2) - 5*Nu + 1)/pow(r, 3);

                else if ((ell == 6) && (mm == 6))
                    HCirc_temp = sqrt(143)*(270.0*pow(Nu, 2) - 270.0*Nu + 54.0)/(715*pow(r, 3));

                HCirc_temp = HCirc_temp*gcmplx::exp(-I*cmplx(mm*phi, 0.0))*sqrt(PI/5)*8.*mu;

                hlms[(bin_i * num_modes + mode_i) * num_steps_max + i] = HCirc_temp;
            }
        }
    }
}

void compute_hlms_wrap(cmplx* hlms, double* r_arr, double* phi_arr, double* pr_arr, double* L_arr,
                  double* m1_arr, double* m2_arr, double* chi1_arr, double* chi2_arr,
                  int* num_steps, int num_steps_max, int* ell_arr_in, int* mm_arr_in, int num_modes, int num_bin_all)
{
    #ifdef __CUDACC__
    compute_hlms<<<num_bin_all, NUM_THREADS_EOB>>>(hlms, r_arr, phi_arr, pr_arr, L_arr,
                  m1_arr, m2_arr, chi1_arr, chi2_arr,
                  num_steps, num_steps_max, ell_arr_in, mm_arr_in, num_modes, num_bin_all);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    #else
    compute_hlms(hlms, r_arr, phi_arr, pr_arr, L_arr,
                  m1_arr, m2_arr, chi1_arr, chi2_arr,
                  num_steps, num_steps_max, ell_arr_in, mm_arr_in, num_modes, num_bin_all);
    #endif
}
