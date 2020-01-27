#CUDAobjects = manager.o interpolate.o likelihood.o kernel.o kernel_response.o createGPUHolders.o
#CPPobjects = fdresponse.o PhenomHM.o IMRPhenomD.o IMRPhenomD_internals.o globalPhenomHM.o RingdownCW.o gpuPhenomHM.o
all: $(CUDAobjects) $(CPPobjects) # tempGPU.o
	#g++ -pthread -shared -fPIC -B /home/mlk667/.conda/envs/gpu4gw_env/compiler_compat -L/home/mlk667/.conda/envs/gpu4gw_env/lib -Wl,-rpath=/home/mlk667/.conda/envs/gpu4gw_env/lib -Wl,--no-as-needed -Wl,--sysroot=/ $(CUDAobjects) $(CPPobjects) tempGPU.o -L/opt/local/lib -L/software/cuda/cuda-9.2/lib64 -Wl,-R/software/cuda/cuda-9.2/lib64 -lcudart -lcublas -lcusparse -lgsl -lgslcblas -lgomp -o gpuPhenomHM.cpython-37m-x86_64-linux-gnu.so

%.o: phenomhm/src/%.cu
	nvcc -arch=sm_70 -Xcompiler -fPIC -I./phenomhm/src/ -I/opt/local/include -I/software/cuda/cuda-9.2/include -dc $< -o $@
%.o: phenomhm/src/%.cpp
	g++ -pthread -B /home/mlk667/.conda/envs/gpu4gw_env/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/mlk667/.conda/envs/gpu4gw_env/lib/python3.7/site-packages/numpy/core/include -I/opt/local/include -I/software/cuda/cuda-9.2/include -Iphenomhm/src -I/home/mlk667/.conda/envs/gpu4gw_env/include/python3.7m -c $< -o $@

gpuPhenomHM.o: phenomhm/gpuPhenomHM.cpp
	g++ -pthread -B /home/mlk667/.conda/envs/gpu4gw_env/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/home/mlk667/.conda/envs/gpu4gw_env/lib/python3.7/site-packages/numpy/core/include -I/opt/local/include -I/software/cuda/cuda-9.2/include -Iphenomhm/src -I/home/mlk667/.conda/envs/gpu4gw_env/include/python3.7m -c $< -o $@

tempGPU.o: $(CUDAobjects)
	nvcc -arch=sm_70 -dlink -Xcompiler -fPIC $(CUDAobjects) -o tempGPU.o

clean:
	rm -f *.o app
