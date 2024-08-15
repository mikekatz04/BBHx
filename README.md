# GPU-Accelerated Black Hole Binary Waveforms (`bbhx`)

### Designed for LISA data analysis of Massive Black Hole Binaries.


This package implements GPU/CPU agnostic Massive Black Hole Binary waveforms and likelihood computations from [arXiv:2005.01827](https://arxiv.org/abs/2005.01827) and [arXiv:2111.01064](https://arxiv.org/abs/2111.01064). The various parts of this package are arranged to be modular as waveform or response changes or improvements are made. Generally, the modules fall into four categories: waveforms, response, waveform building, and utilities. Please see the [documentation](https://mikekatz04.github.io/BBHx/) for further information on these modules. The code can be found on Github [here](https://github.com/mikekatz04/BBHx).

This package is a part of the LISA Analysis Tools environment.

If you use this software please cite [arXiv:2005.01827](https://arxiv.org/abs/2005.01827), [arXiv:2111.01064](https://arxiv.org/abs/2111.01064), and the associated [Zenodo page](https://zenodo.org/record/5730688#.YaFvRkJKhTY) Please also cite any consituent parts used like the response function or waveforms. See the `citation` attribute for each class or docstring for functions for more information.

## Getting Started

To install with pip (for CPUs only currently):
```
pip install bbhx
```

2) To import ``bbhx``:

```
from bbhx.waveform import BBHWaveformFD
```

See [examples notebook](https://github.com/mikekatz04/BBHx/blob/master/examples/bbhx_tutorial.ipynb).


### Prerequisites

To install this software for CPU usage, you need [gsl >2.0](https://www.gnu.org/software/gsl/) , [lapack (3.6.1)](https://www.netlib.org/lapack/lug/node14.html), Python >3.4, and NumPy. If you install lapack with conda, the new version (3.9) seems to not install the correct header files. Therefore, the lapack version must be 3.6.1. To run the examples, you will also need jupyter and matplotlib. We generally recommend installing everything, including gcc and g++ compilers, in the conda environment as is shown in the examples here. This generally helps avoid compilation and linking issues. If you use your own chosen compiler, you will need to make sure all necessary information is passed to the setup command (see below). You also may need to add information to the `setup.py` file.

To install this software for use with NVIDIA GPUs (compute capability >2.0), you need the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CuPy](https://cupy.chainer.org/). The CUDA toolkit must have cuda version >8.0. Be sure to properly install CuPy within the correct CUDA toolkit version. Make sure the nvcc binary is on `$PATH` or set it as the `CUDAHOME` environment variable.

### Installing

To install with pip (for CPUs only currently):
```
pip install bbhx
```

To install from source:

0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Clone the repository.

```
git clone https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms.git
cd FastEMRIWaveforms
```

2) Installation is made easy through [install.sh](install.sh). This is a bash script that will create a conda environment, install bbhx, run tests, and install any additional packages needed for sampling or development. It will look for an `nvcc` binary, the `CUDA_HOME` variable, or the `CUDAHOME` variable. If it finds that information, it will install for CUDA as well (including installing the proper version of `cupy`). **Note**: If you already have performed installation and you are updating bbhx after a `git pull`, then run `pip install .` rather than the following command.

  ```
  bash install.sh
  ```

  Options for installation can be applied by running `bash install.sh key=value`. These can be found with `bash install.sh -h`:
  
  ```
  keyword argument options (given as key=value):
    env_name:  Name of generated conda environment. Default is 'bbhx_env'.
    install_type:  Type of install. 'basic', 'development', or 'sampling'. 
        'development' adds packages needed for development and documentation.
        'sampling' adds packages for sampling like eryn, lisatools, corner, chainconsumer.
        Default is 'basic'. 
    run_tests: Either true or false. Whether to run tests after install. Default is true.
  ```

3) Load the environment (change "bbhx_env" to the correct environment name is specified in previous step):

```
conda activate bbhx_env
```

Please contact the developers if the installation does not work.


### More Customized Install (legacy)
0) [Install Anaconda](https://docs.anaconda.com/anaconda/install/) if you do not have it.

1) Create a virtual environment.

```
conda create -n bbhx_env -c conda-forge gcc_linux-64 gxx_linux-64 gsl lapack=3.6.1 numpy scipy Cython jupyter ipython matplotlib python=3.9
conda activate bbhx_env
```

    If on MACOSX, substitute `gcc_linux-64` and `gxx_linus-64` with `clang_osx-64` and `clangxx_osx-64`.

    If you want a faster install, you can install the python packages (numpy, Cython, jupyter, ipython, matplotlib) with pip.

2) Clone the repository.

```
git clone https://mikekatz04.github.io/BBHx.git
cd BBHx
```

3) If using GPUs, use pip to [install cupy](https://docs-cupy.chainer.org/en/stable/install.html). If you have cuda version 9.2, for example:

```
pip install cupy-cuda92
```

4) Run install. Make sure CUDA is on your PATH or `CUDAHOME` variable is set to the path to nvcc and other CUDA files.

```
python setup.py install
```


When installing lapack and gsl, the setup file will default to assuming lib and include for both are in installed within the conda environment. To provide other lib and include directories you can provide command line options when installing. You can also remove usage of OpenMP.

```
python setup.py --help
usage: setup.py [-h] [--no_omp] [--lapack_lib LAPACK_LIB]
                [--lapack_include LAPACK_INCLUDE] [--lapack LAPACK]
                [--gsl_lib GSL_LIB] [--gsl_include GSL_INCLUDE] [--gsl GSL]
                [--ccbin CCBIN]

optional arguments:
  -h, --help            show this help message and exit
  --no_omp              If provided, install without OpenMP.
  --lapack_lib LAPACK_LIB
                        Directory of the lapack lib. If you add lapack lib,
                        must also add lapack include.
  --lapack_include LAPACK_INCLUDE
                        Directory of the lapack include. If you add lapack
                        includ, must also add lapack lib.
  --lapack LAPACK       Directory of both lapack lib and include. '/include'
                        and '/lib' will be added to the end of this string.
  --gsl_lib GSL_LIB     Directory of the gsl lib. If you add gsl lib, must
                        also add gsl include.
  --gsl_include GSL_INCLUDE
                        Directory of the gsl include. If you add gsl include,
                        must also add gsl lib.
  --gsl GSL             Directory of both gsl lib and include. '/include' and
                        '/lib' will be added to the end of this string.
  --ccbin CCBIN         path/to/compiler to link with nvcc when installing
                        with CUDA.
```



## Running the Tests

In the main directory of the package run in the terminal:
```
python -m unittest discover
```


## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/BBHx/tags).

Current Version: 1.1.0

## Authors

* **Michael Katz**

### Contibutors

* Sylvain Marsat
* John Baker

## License

This project is licensed under the GNU License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

* This research was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
