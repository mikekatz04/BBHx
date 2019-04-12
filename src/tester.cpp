#include <stdlib.h>
#include <complex>


void double_errthing(int * arr_in, int length){
    int i;
    
    std::complex<double> comp_test = 1.0 + 1.0*I;
    for (i=0; i<length; i++) arr_in[i] = 2.*arr_in[i];
}
