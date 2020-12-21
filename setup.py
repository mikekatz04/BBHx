# from future.utils import iteritems
import os
import sys
from os.path import join as pjoin
from setuptools import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy
import shutil
import argparse


def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system

    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.

    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if "CUDAHOME" in os.environ:
        home = os.environ["CUDAHOME"]
        nvcc = pjoin(home, "bin", "nvcc")
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path("nvcc", os.environ["PATH"])
        if nvcc is None:
            raise EnvironmentError(
                "The nvcc binary could not be "
                "located in your $PATH. Either add it to your path, "
                "or set $CUDAHOME"
            )
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {
        "home": home,
        "nvcc": nvcc,
        "include": pjoin(home, "include"),
        "lib64": pjoin(home, "lib64"),
    }
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError(
                "The CUDA %s path could not be " "located in %s" % (k, v)
            )

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.

    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append(".cu")

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        if os.path.splitext(src)[1] == ".cu":
            # use the cuda for .cu files
            self.set_executable("compiler_so", CUDA["nvcc"])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs["nvcc"]
        else:
            postargs = extra_postargs["gcc"]

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile


# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)


try:
    CUDA = locate_cuda()
    run_cuda_install = True
except OSError:
    run_cuda_install = False

parser = argparse.ArgumentParser()

parser.add_argument(
    "--no_omp",
    help="If provided, install without OpenMP.",
    action="store_true",
    default=False,
)

parser.add_argument(
    "--lapack_lib",
    help="Directory of the lapack lib.",
    default="/usr/local/opt/lapack/lib",
)

parser.add_argument(
    "--lapack_include",
    help="Directory of the lapack include.",
    default="/usr/local/opt/lapack/include",
)

parser.add_argument(
    "--lapack",
    help="Directory of both lapack lib and include. '/include' and '/lib' will be added to the end of this string.",
)

parser.add_argument(
    "--gsl_lib", help="Directory of the gsl lib.", default="/usr/local/opt/gsl/lib"
)

parser.add_argument(
    "--gsl_include",
    help="Directory of the gsl include.",
    default="/usr/local/opt/gsl/include",
)

parser.add_argument(
    "--gsl",
    help="Directory of both gsl lib and include. '/include' and '/lib' will be added to the end of this string.",
)

args, unknown = parser.parse_known_args()

for key in [
    args.gsl_include,
    args.gsl_lib,
    args.gsl,
    "--gsl",
    "--gsl_include",
    "--gsl_lib",
    args.lapack_include,
    args.lapack_lib,
    args.lapack,
    "--lapack",
    "--lapack_lib",
    "--lapack_include",
]:
    try:
        sys.argv.remove(key)
    except ValueError:
        pass

use_omp = not args.no_omp


# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

if args.lapack is None:
    lapack_include = [args.lapack_include]
    lapack_lib = [args.lapack_lib]

else:
    lapack_include = [args.lapack + "/include"]
    lapack_lib = [args.lapack + "/lib"]

if args.gsl is None:
    gsl_include = [args.gsl_include]
    gsl_lib = [args.gsl_lib]

else:
    gsl_include = [args.gsl + "/include"]
    gsl_lib = [args.gsl + "/lib"]


if "--no_omp" in sys.argv:
    use_omp = False
    sys.argv.remove("--no_omp")

else:
    use_omp = True


# if installing for CUDA, build Cython extensions for gpu modules
if run_cuda_install:

    gpu_extension = dict(
        libraries=["cudart", "cublas", "cusparse", "gsl", "gslcblas", "gomp"],
        library_dirs=[CUDA["lib64"]] + gsl_lib,
        runtime_library_dirs=[CUDA["lib64"]],
        language="c++",
        # This syntax is specific to this build system
        # we're only going to use certain compiler args with nvcc
        # and not with gcc the implementation of this trick is in
        # customize_compiler()
        extra_compile_args={
            "gcc": ["-std=c++11", "-fopenmp", "-D__USE_OMP__"],  # '-g'],
            "nvcc": [
                "-arch=sm_70",
                # "-gencode=arch=compute_30,code=sm_30",
                # "-gencode=arch=compute_50,code=sm_50",
                # "-gencode=arch=compute_52,code=sm_52",
                # "-gencode=arch=compute_60,code=sm_60",
                # "-gencode=arch=compute_61,code=sm_61",
                "-gencode=arch=compute_70,code=sm_70",
                #'-gencode=arch=compute_75,code=sm_75',
                #'-gencode=arch=compute_75,code=compute_75',
                "-std=c++11",
                "--default-stream=per-thread",
                "--ptxas-options=-v",
                "-c",
                "--compiler-options",
                "'-fPIC'",
                "-Xcompiler",
                "-fopenmp",
                "-D__USE_OMP__",
                # "-G",
                # "-g",
                # "-O0",
                # "-lineinfo",
            ],  # for debugging
        },
        include_dirs=[numpy_include, CUDA["include"], "include"] + gsl_include,
    )

    if use_omp is False:
        gpu_extension["extra_compile_args"]["nvcc"].remove("-fopenmp")
        gpu_extension["extra_compile_args"]["gcc"].remove("-fopenmp")
        gpu_extension["extra_compile_args"]["nvcc"].remove("-D__USE_OMP__")
        gpu_extension["extra_compile_args"]["gcc"].remove("-D__USE_OMP__")

    pyHdynBBH_ext = Extension(
        "pyHdynBBH", sources=["src2/full.cu", "src2/hdyn.pyx"], **gpu_extension
    )

    # gpu_extensions.append(Extension(extension_name, **temp_dict))

"""
# shutil.copy("phenomhm/gpuPhenomHM.pyx", "phenomhm/cpuPhenomHM.pyx")
# shutil.copy("phenomd/gpuPhenomD.pyx", "phenomd/cpuPhenomD.pyx")
# Obtain the numpy include directory. This logic works across numpy versions.
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

print("\n\n\n\n\nOn quest:", os.path.isdir("/home/mlk667/GPU4GW"), "\n\n\n\n\n")
if os.path.isdir("/home/mlk667/GPU4GW"):
    lapack_include = ["/software/lapack/3.6.0_gcc/include/"]
    lapack_lib = ["/software/lapack/3.6.0_gcc/lib64/"]

else:
    lapack_include = ["/usr/local/opt/lapack/include"]
    lapack_lib = ["/usr/local/opt/lapack/lib"]

print(lapack_include)

lib_gsl_dir = "/opt/local/lib"
include_gsl_dir = "/opt/local/include"

cpu_libs = ["pthread", "lapack"]

cpu_extra_compile_args = {"gcc": ["-O3", "-fopenmp", "-fPIC"]}

cpu_extra_link_args = ["-Wl,-rpath,/usr/local/opt/gcc/lib/gcc/9/"]

temp_files = []
for i, source in enumerate(phenomhm_sources):
    temp = os.path.splitext(source)[0]
    file_ext = os.path.splitext(source)[1]
    if file_ext == ".cu":
        shutil.copy(temp + ".cu", temp + ".cpp")
        temp_files.append(temp + ".cpp")
        phenomhm_sources[i] = temp + ".cpp"

        if temp + ".cu" in phenomd_sources:
            ind = phenomd_sources.index(temp + ".cu")
            phenomd_sources[ind] = temp + ".cpp"

for i, source in enumerate(phenomd_sources):
    temp = os.path.splitext(source)[0]
    file_ext = os.path.splitext(source)[1]

    if temp + ".cpp" in temp_files:
        continue
    if file_ext == ".cu":
        shutil.copy(temp + ".cu", temp + ".cpp")
        temp_files.append(temp + ".cpp")
        phenomd_sources[i] = temp + ".cpp"

cpu_extension_dict = dict(
    sources=phenomhm_sources,
    library_dirs=all_lib_dirs + lapack_lib,
    libraries=all_libs + cpu_libs,
    language="c++",
    # This syntax is specific to this build system
    # we're only going to use certain compiler args with nvcc
    # and not with gcc the implementation of this trick is in
    # customize_compiler()
    extra_compile_args=cpu_extra_compile_args,
    extra_link_args=cpu_extra_link_args,
    include_dirs=all_include + lapack_include,
)

cpu_extensions = []

temp_dict = copy.deepcopy(cpu_extension_dict)
extension_name = "cpuPhenomHM"
folder = "phenomhm/"
temp_dict["sources"] += [folder + extension_name + ".pyx"]

cpu_extensions.append(Extension(extension_name, **temp_dict))

shutil.copy(folder + extension_name + ".pyx", folder + extension_name + "_glob.pyx")

temp_dict = copy.deepcopy(cpu_extension_dict)
extension_name = "cpuPhenomHM_glob"
folder = "phenomhm/"
temp_dict["sources"] += [folder + extension_name + ".pyx"]
temp_dict["extra_compile_args"]["gcc"].append("-D__GLOBAL_FIT__")

cpu_extensions.append(Extension(extension_name, **temp_dict))

temp_dict = copy.deepcopy(cpu_extension_dict)
extension_name = "cpuPhenomD"
folder = "phenomd/"
temp_dict["sources"] = phenomd_sources + [folder + extension_name + ".pyx"]

cpu_extensions.append(Extension(extension_name, **temp_dict))

shutil.copy(folder + extension_name + ".pyx", folder + extension_name + "_glob.pyx")

temp_dict = copy.deepcopy(cpu_extension_dict)
extension_name = "cpuPhenomD_glob"
folder = "phenomd/"
temp_dict["sources"] = phenomd_sources + [folder + extension_name + ".pyx"]
temp_dict["extra_compile_args"]["gcc"].append("-D__GLOBAL_FIT__")

cpu_extensions.append(Extension(extension_name, **temp_dict))
"""

if run_cuda_install:
    extensions = [pyHdynBBH_ext]

setup(
    name="hdyn",
    author="Michael Katz",
    author_email="mikekatz04@gmail.com",
    ext_modules=extensions,
    packages=["phenomhm", "phenomhm.utils"],
    # Inject our custom trigger
    cmdclass={"build_ext": custom_build_ext},
    # Since the package has c code, the egg cannot be zipped
    zip_safe=False,
    python_requires=">=3.6",
)
