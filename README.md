# GPU-Accelerated Black Hole Binary Waveforms (`bbhx`)

**Designed for LISA data analysis of Massive Black Hole Binaries.**


This package implements GPU/CPU agnostic Massive Black Hole Binary waveforms and likelihood computations from [arXiv:2005.01827](https://arxiv.org/abs/2005.01827) and [arXiv:2111.01064](https://arxiv.org/abs/2111.01064). The various parts of this package are arranged to be modular as waveform or response changes or improvements are made. The main modules are:

- frequency-domain waveforms (PhenomHM / PhenomD, `bbhx.waveforms.phenomhm`);
- the fast frequency-domain LISA response (`bbhx.response.fastfdresponse`);
- waveform building (`bbhx.waveformbuild.BBHWaveformFD`);
- likelihoods (`bbhx.likelihood`: direct and heterodyned);
- time-domain TDI-on-the-fly generators for MBHs (`bbhx.mbhtdionfly`, using the external [phentax](https://github.com/asantini29/phentax) model) and for stellar-origin BBHs (`bbhx.sobbhtdionfly`);
- the SOBBH chunked-heterodyne WDM likelihood (`bbhx.sobbhcomps`) and its native kernels;
- utilities (SSB <-> LISA frame transforms, constants, backends).

Please see the [documentation](https://lisa-analysis-tools.github.io/BBHx/) for further information on these modules. The code can be found on Github [here](https://github.com/lisa-analysis-tools/BBHx).

This package is a part of the LISA Analysis Tools environment. Generic LISA infrastructure (orbits, TDI configuration, the time-domain response, WDM / FD domains, sensitivity curves) comes from [LISA Analysis Tools](https://github.com/lisa-analysis-tools/lisa-analysis-tools) (`lisaanalysistools`), and the backend machinery from `gpubackendtools`. Physical constants come from `lisaconstants`.

If you use this software please cite [arXiv:2005.01827](https://arxiv.org/abs/2005.01827), [arXiv:2111.01064](https://arxiv.org/abs/2111.01064), and the associated [Zenodo page](https://zenodo.org/records/17195311). Please also cite any constituent parts used, like the response function or waveforms. The papers for each class are listed in its docstring. (The classes' `citation` properties currently raise `NameError`: the citation module they read was removed.)


## Getting started

Detailed installation instructions can be found in the [documentation](https://lisa-analysis-tools.github.io/BBHx/).
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

The CUDA 13 backend (`cuda13x`) is not published as a wheel. A source build with a
CUDA 13 toolkit builds it (see "Installing from sources" below).

To use the time-domain MBH TDI-on-the-fly generator
(`bbhx.mbhtdionfly.MBHTDIonFly`), you also need the
[phentax](https://github.com/asantini29/phentax) waveform package
(IMRPhenomTHM). It is not published on PyPI, so install it through the
`phentax` extra (which pulls it from GitHub) or directly:

```sh
# Via the extra (from a source/git install of bbhx)
pip install 'bbhx[phentax]'

# Or directly
pip install git+https://github.com/asantini29/phentax.git
```

To know your CUDA version, run the tool `nvidia-smi` in a terminal and check the CUDA version reported in the table header:

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
>>> for backend in ["cpu", "cuda11x", "cuda12x", "cuda13x"]:
...     status = "available" if bbhx.has_backend(backend) else "unavailable"
...     print(f" - Backend '{backend}': {status}")
 - Backend 'cpu': available
 - Backend 'cuda11x': unavailable
 - Backend 'cuda12x': unavailable
 - Backend 'cuda13x': unavailable
```

`bbhx.has_backend` accepts these flavor names (or the full `bbhx_<flavor>` names). It
does not accept the `cuda` / `gpu` aliases: it raises `ValueError` for them. For
`bbhx.get_backend` and for the `force_backend` argument of the compute classes, `cuda`
(and its alias `gpu`) means the first available of `cuda13x`, `cuda12x` and `cuda11x`.

If you expected a backend to be available but it is not, call `bbhx.get_backend` with its
name, e.g. `bbhx.get_backend("cuda12x")`. The exception it raises is chained to the
original import error, which names the missing piece. A BBHx backend needs three native
modules of the same flavor (`bbhx_backend_<flavor>`, `lisatools_backend_<flavor>` and
`gbt_backend_<flavor>`), and a CUDA backend also needs `cupy`.

Once BBHx is working and the expected backends are selected, check out the [examples notebooks](https://github.com/lisa-analysis-tools/BBHx/tree/master/examples/)
on how to start with this software. The LISA Analysis Tools Workshop tutorials
([LATW](https://github.com/lisa-analysis-tools/LATW)) cover the wider stack.

## Installing from sources

### Prerequisites

To install this software from source, you will need:

- A C++ compiler (g++, clang++, ...)
- Python 3.12 or newer. BBHx itself declares `>=3.9`, but its `lisaanalysistools`
  dependency requires `>=3.12`.
- The build requirements listed in `pyproject.toml`, in particular `gpubackendtools`,
  `lisaanalysistools` and `nanobind==2.12.0` (pip installs them in its isolated build
  environment unless you pass `--no-build-isolation`). The build compiles sources and
  headers taken from the installed `lisaanalysistools` (e.g. `Detector.cu`,
  `lat_tdi_on_the_fly.cu`) and `gpubackendtools` (`Interpolate.cu`) packages into
  BBHx's native module.

Some installation steps require the external library `LAPACK` along with its C-bindings provided by `LAPACKE`.
If these libraries and their header files (in particular `lapacke.h`) are available on your system, they will be detected
and used automatically. Detection is delegated to GPUBackendTools' `get_lapacke()` and
is controlled by the CMake options `GBT_LAPACKE_DETECT_WITH` (`AUTO`, `CMAKE`,
`PKGCONFIG` or `DISABLE`), `GBT_LAPACKE_FETCH` (`AUTO`, `ON` or `OFF`) and
`GBT_LAPACKE_EXTRA_LIBS` (e.g. `gfortran`). Pass them as
`pip install --config-settings=cmake.define.GBT_LAPACKE_DETECT_WITH=PKGCONFIG .`.
By default, if `LAPACKE` is not found on your system, the installation step downloads its sources
and adds them to the compilation tree. This makes the installation a bit longer but a lot easier.

If you want to enable GPU support in BBHx, you will also need the NVIDIA CUDA Compiler `nvcc` in your path as well as
the [CUDA toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) (with, in particular, the
libraries `CUDA Runtime Library`, `cuBLAS` and `cuSPARSE`). The CMake option `BBHX_WITH_GPU` controls the GPU build:

- `AUTO` (default): build the GPU backend when a CUDA toolchain is found;
- `ON`: build it, and fail without a toolchain;
- `OFF`: build the CPU backend only;
- `ONLY`: build the GPU backend only;
- `BARE`: build no backend.

`BBHX_CUDA_ARCH` sets the CUDA architectures (default `70;75;80;86;90`; `70` is dropped
for CUDA >= 13). The GPU backend is named after the toolkit's major version
(`cuda11x`, `cuda12x` or `cuda13x`).

### Installation instructions using conda

We recommend installing BBHx using conda, so that the compilers are all within one environment.
First clone the repo

```
git clone https://github.com/lisa-analysis-tools/BBHx.git
cd BBHx
```

Now create an environment (these instructions work for all platforms, but some
adjustments may be needed):

```
conda create -n bbhx_env -y -c conda-forge --override-channels \
    python=3.12 cxx-compiler pkgconfig conda-forge/label/lapack_rc::liblapacke
```

activate the environment

```
conda activate bbhx_env
```

Then we can install locally for development:
```
pip install -e '.[doc,testing]'
```

### The development stack

On the `dev` branch, BBHx is built against the development branches of the LISA
Analysis Tools packages (GPUBackendTools, LISA Analysis Tools, ...). The central
installer `install.sh` of the
[LISA Analysis Tools repository](https://github.com/lisa-analysis-tools/lisa-analysis-tools)
clones and installs the whole stack side by side. It installs BBHx (branch `dev`) with
`pip install --no-build-isolation -e .`, so that the build uses the
already installed development versions of `gpubackendtools` and `lisaanalysistools`.

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
git clone https://github.com/lisa-analysis-tools/BBHx.git
cd BBHx

# Run the install
pip install .
```

If the installation does not work, first check the [detailed installation
documentation](https://lisa-analysis-tools.github.io/BBHx/). If
it still does not work, please open an issue on the
[GitHub repository](https://github.com/lisa-analysis-tools/BBHx/issues)
or contact the developers through other means.



### Running the Tests

The tests need some dependencies that are not installed by default (the `[testing]`
extra adds `matplotlib`). The MBH TDI-on-the-fly waveform test also needs `phentax` and
is skipped without it. To install the extra, add the `[testing]` label to the BBHx
package name when installing it. E.g:

```sh
# For CPU-only version with testing enabled
pip install bbhx[testing]

# For GPU version with CUDA 12.Y and testing enabled
pip install bbhx-cuda12x[testing]

# For from-source install with testing enabled
git clone https://github.com/lisa-analysis-tools/BBHx.git
cd BBHx
pip install '.[testing]'
```

To run the tests, open a terminal in a directory containing the sources of BBHx and then run the `unittest` module in `discover` mode:

```sh
$ git clone https://github.com/lisa-analysis-tools/BBHx.git
$ cd BBHx
$ python -m unittest discover  # the CI runs "python -m unittest tests/test*"
...
OK
```

## Contributing

Please read [CONTRIBUTING.md](https://github.com/lisa-analysis-tools/BBHx/blob/dev/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

If you want to develop BBHx and produce documentation, install `bbhx` from source with the `[doc]` label and in `editable` mode:

```
$ git clone https://github.com/lisa-analysis-tools/BBHx.git
$ cd BBHx
pip install -e '.[doc,testing]'
```

This will install the packages needed to build the documentation (`sphinx`, `sphinx-rtd-theme`, `sphinx-tippy`, `myst-parser`, `nbsphinx`, `pypandoc`, `ipython`, `ipykernel`, `ipywidgets`) and to run the tests.

The documentation source files are in `docs/source`. To compile the documentation locally, change to the `docs` directory and run `make html`. The tutorial notebooks are copied from `examples/` at build time. Some API pages import optional packages: `jax` for the JAX SOBBH source, and `eryn` (the `[sampling]` extra) for `bbhx.mbhphentax`.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/lisa-analysis-tools/BBHx/tags).

## Contributors

A (non-exhaustive) list of contributors can be found in [CONTRIBUTORS.md](https://github.com/lisa-analysis-tools/BBHx/blob/dev/CONTRIBUTORS.md).

## License

This project is licensed under the GNU General Public License, version 3 or (at your option) any later version - see the [LICENSE](https://github.com/lisa-analysis-tools/BBHx/blob/dev/LICENSE) file for details.

## Citation

Please cite [arXiv:2005.01827](https://arxiv.org/abs/2005.01827), [arXiv:2111.01064](https://arxiv.org/abs/2111.01064) and the BBHx software on [Zenodo](https://zenodo.org/records/17195311).
Other papers need to be cited depending on the classes used (waveform models, response, heterodyning). They are listed in each class's docstring.

## Acknowledgments

* This research resulting in this code was supported by National Science Foundation under grant DGE-0948017 and the Chateaubriand Fellowship from the Office for Science \& Technology of the Embassy of France in the United States.
* It was also supported in part through the computational resources and staff contributions provided for the Quest/Grail high performance computing facility at Northwestern University.
