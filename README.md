# GPU-Accelerated Black Hole Binary Waveforms (`bbhx`)

### Designed for LISA data analysis of Massive Black Hole Binaries.


This package implements GPU/CPU agnostic Massive Black Hole Binary waveforms and likelihood computations from [arXiv:2005.01827](https://arxiv.org/abs/2005.01827) and [arXiv:2111.01064](https://arxiv.org/abs/2111.01064). The various parts of this package are arranged to be modular as waveform or response changes or improvements are made. Generally, the modules fall into four categories: waveforms, response, waveform building, and utilities. Please see the [documentation](https://mikekatz04.github.io/BBHx/) for further information on these modules. The code can be found on Github [here](https://github.com/mikekatz04/BBHx).

This package is a part of the LISA Analysis Tools environment.

If you use this software please cite [arXiv:2005.01827](https://arxiv.org/abs/2005.01827), [arXiv:2111.01064](https://arxiv.org/abs/2111.01064), and the associated [Zenodo page](https://zenodo.org/records/17195311) Please also cite any consituent parts used like the response function or waveforms. See the `citation` attribute for each class or docstring for functions for more information.


## Getting started

Detailed installation instructions can be found in the [documentation](https://mikekatz04.github.io/BBHx/).
Below is a quick set of instructions to install the BBHx package on CPUs and GPUs.

To install the latest version of `bbhx` using `pip`, simply run:

```sh
# For CPU-only version
pip install bbhx

# For GPU-enabled versions with CUDA 11.Y.Z
pip install bbhx-cuda11x

# For GPU-enabled versions with CUDA 12.Y.Z
pip install bbhx-cuda12x
```

To know your CUDA version, run the tool `nvidia-smi` in a terminal a check the CUDA version reported in the table header:

```sh
$ nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15              Driver Version: 550.54.15      CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
...
```

You may also install `bbhx` directly using conda (including on Windows)
as well as its CUDA 12.x plugin (only on Linux). It is strongly advised to:

1. Ensure that your conda environment makes sole use of the `conda-forge` channel
2. Install `bbhx` directly when building your conda environment, not afterwards

```sh
# For CPU-only version, on either Linux, macOS or Windows:
conda create --name bbhx_cpu -c conda-forge --override-channels python=3.12 bbhx
conda activate bbhx_cpu

# For CUDA 12.x version, only on Linux
conda create --name bbhx_cuda -c conda-forge --override-channels python=3.12 bbhx-cuda12x
conda activate bbhx_cuda
```

Now, in a python file or notebook:

```py3
import bbhx
```

You may check the currently available backends:

```py3
>>> for backend in ["cpu", "cuda11x", "cuda12x", "cuda", "gpu"]:
...     print(f" - Backend '{backend}': {"available" if bbhx.has_backend(backend) else "unavailable"}")
 - Backend 'cpu': available
 - Backend 'cuda11x': unavailable
 - Backend 'cuda12x': unavailable
 - Backend 'cuda': unavailable
 - Backend 'gpu': unavailable
```

Note that the `cuda` backend is an alias for either `cuda11x` or `cuda12x`. If any is available, then the `cuda` backend is available.
Similarly, the `gpu` backend is (for now) an alias for `cuda`.

If you expected a backend to be available but it is not, run the following command to obtain an error
message which can guide you to fix this issue:

```py3
>>> import bbhx
>>> bbhx.get_backend("cuda12x")
ModuleNotFoundError: No module named 'bbhx_backend_cuda12x'

The above exception was the direct cause of the following exception:
...

bbhx.cutils.BackendNotInstalled: The 'cuda12x' backend is not installed.

The above exception was the direct cause of the following exception:
...

bbhx.cutils.MissingDependencies: BBHx CUDA plugin is missing.
    If you are using bbhx in an environment managed using pip, run:
        $ pip install bbhx-cuda12x

The above exception was the direct cause of the following exception:
...

bbhx.cutils.BackendAccessException: Backend 'cuda12x' is unavailable. See previous error messages.
```

Once FEW is working and the expected backends are selected, check out the [examples notebooks](https://github.com/mikekatz04/BBHx/tree/master/examples/)
on how to start with this software.

## Installing from sources

### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- A Python version supported by [scikit-build-core](https://github.com/scikit-build/scikit-build-core) (>=3.7 as of Jan. 2025)

Some installation steps require the external library `LAPACK` along with its C-bindings provided by `LAPACKE`.
If these libraries and their header files (in particular `lapacke.h`) are available on your system, they will be detected
and used automatically. If they are available on a non-standard location, see below for some options to help detecting them.
Note that by default, if `LAPACKE` is not available on your system, the installation step will attempt to download its sources
and add them to the compilation tree. This makes the installation a bit longer but a lot easier.

If you want to enable GPU support in FEW, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`).

There are a set of files required for total use of this package. They will download automatically the first time they are needed. Files are generally under 10MB. However, there is a 100MB file needed for the slow waveform and the bicubic amplitude interpolation. This larger file will only download if you run either of those two modules. The files are hosted on the [Black Hole Perturbation Toolkit Download Server](https://download.bhptoolkit.org/bbhx/data).

### Installation instructions using conda

We recommend to install FEW using conda in order to have the compilers all within an environment.
First clone the repo

```
git clone https://github.com/mikekatz04/BBHx.git
cd BBHx
```

Now create an environment (these instructions work for all platforms but some
adjustements can be needed, refer to the
[detailed installation documentation](https://bbhx.readthedocs.io/en/stable/user/install.html) for more information):

```
conda create -n bbhx_env -y -c conda-forge --override-channels |
    cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

activate the environment

```
conda activate bbhx_env
```

Then we can install locally for development:
```
pip install -e '.[dev, testing]'
```

### Installation instructions using conda on GPUs and linux
Below is a quick set of instructions to install the BBHx package on GPUs and linux.

```sh
conda create -n bbhx_env -c conda-forge bbhx-cuda12x python=3.12
conda activate bbhx_env
```

Test the installation device by running python
```python
import bbhx
bbhx.get_backend("cuda12x")
```

### Running the installation

To start the from-source installation, ensure the pre-requisite are met, clone
the repository, and then simply run a `pip install` command:

```sh
# Clone the repository
git clone https://github.com/mikekatz04/BBHx.git
cd BBHx

# Run the install
pip install .
```

If the installation does not work, first check the [detailed installation
documentation](https://mikekatz04.github.io/BBHx/). If
it still does not work, please open an issue on the
[GitHub repository](https://github.com/mikekatz04/BBHx/issues)
or contact the developers through other means.



### Running the Tests

The tests require a bbhx dependencies which are not installed by default. To install them, add the `[testing]` label to FEW package
name when installing it. E.g:

```sh
# For CPU-only version with testing enabled
pip install bbhx[testing]

# For GPU version with CUDA 12.Y and testing enabled
pip install bbhx-cuda12x[testing]

# For from-source install with testing enabled
git clone https://github.com/mikekatz04/BBHx.git
cd BBHx
pip install '.[testing]'
```

To run the tests, open a terminal in a directory containing the sources of FEW and then run the `unittest` module in `discover` mode:

```sh
$ git clone https://github.com/mikekatz04/BBHx.git
$ cd BBHx
$ python -m bbhx.tests  # or "python -m unittest discover"
...
----------------------------------------------------------------------
Ran 20 tests in 71.514s
OK
```

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

If you want to develop FEW and produce documentation, install `bbhx` from source with the `[dev]` label and in `editable` mode:

```
$ git clone https://github.com/mikekatz04/BBHx.git
$ cd BBHx
pip install -e '.[dev, testing]'
```

This will install necessary packages for building the documentation (`sphinx`, `pypandoc`, `sphinx_rtd_theme`, `nbsphinx`) and to run the tests.

The documentation source files are in `docs/source`. To compile the documentation locally, change to the `docs` directory and run `make html`.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/mikekatz04/BBHx/tags).

## Contributors

A (non-exhaustive) list of contributors to the FEW code can be found in [CONTRIBUTORS.md](CONTRIBUTORS.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

Please make sure to cite FEW papers and the FEW software on [Zenodo](https://zenodo.org/records/3969004).
We provide a set of prepared references in [PAPERS.bib](PAPERS.bib). There are other papers that require citation based on the classes used. For most classes this applies to, you can find these by checking the `citation` attribute for that class.  All references are detailed in the [CITATION.cff](CITATION.cff) file.

## Acknowledgments

* This research resulting in this code was supported by National Science Foundation under grant DGE-0948017 and the Chateaubriand Fellowship from the Office for Science \& Technology of the Embassy of France in the United States.
* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
