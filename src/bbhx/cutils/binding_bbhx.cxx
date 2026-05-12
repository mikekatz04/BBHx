#include "PhenomHMWaveform.hh"
#include "Response.hh"
#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "binding_bbhx.hpp"
#include "gbt_global.h"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#include "pybind11_cuda_array_interface.hpp"
#endif

namespace py = pybind11;



std::string get_module_path_cbbhx() {
    // Acquire the GIL if it's not already held (safe to call multiple times)
    py::gil_scoped_acquire acquire;

    // Import the module by its name
    // Note: The module name here ("cbbhx") must match the name used in PYBIND11_MODULE
    py::object module = py::module::import("cbbhx");

    // Access the __file__ attribute and cast it to a C++ string
    try {
        std::string path = module.attr("__file__").cast<std::string>();
        return path;
    } catch (const py::error_already_set& e) {
        // Handle the error if __file__ attribute is missing (e.g., if module is a namespace package)
        std::cerr << "Error getting __file__ attribute: " << e.what() << std::endl;
        return "";
    }
}


// PYBIND11_MODULE creates the entry point for the Python module
// The module name here must match the one used in CMakeLists.txt
void response_part(py::module &m) {

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<BBHxComputationWrap>(m, "BBHxComputationWrapGPU")
#else
    py::class_<BBHxComputationWrap>(m, "BBHxComputationWrapCPU")
#endif 

    // Bind the constructor
    .def(py::init<>())
    // .def(py::init<OrbitsWrap_bbhx *>(), 
    //      py::arg("orbits"))
    // Bind member functions
    .def("get_phenomhm_ringdown_frequencies", &BBHxComputationWrap::get_phenomhm_ringdown_frequencies, "PhenomHM Ringdown frequencies.")
    .def("get_phenomd_ringdown_frequencies", &BBHxComputationWrap::get_phenomd_ringdown_frequencies, "PhenomD Ringdown frequencies.")
    .def("waveform_amp_phase_wrap", &BBHxComputationWrap::waveform_amp_phase_wrap, "PhenomHM Amp/Phase.")
    .def("LISA_response", &BBHxComputationWrap::LISA_response, "LISA response function.")
    .def("InterpTDI_wrap", &BBHxComputationWrap::InterpTDI_wrap, "InterpTDI function.")
    .def("hdyn_wrap", &BBHxComputationWrap::hdyn_wrap, "Heterodyne in FD.")
    .def("direct_like_wrap", &BBHxComputationWrap::direct_like_wrap, "Direct Likelihood.")
    .def("direct_sum_wrap", &BBHxComputationWrap::direct_sum_wrap, "Direct waveform summation.")
    .def("prep_hdyn", &BBHxComputationWrap::prep_hdyn, "Prepare Hdyn coefficients.")
    .def("interpolate_wrap", &BBHxComputationWrap::interpolate_wrap, "Interpolation.")
    
    // .def_readwrite("orbits", &BBHxComputationWrap::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<OrbitsWrap_bbhx>(m, "OrbitsWrapGPU_bbhx")
#else
    py::class_<OrbitsWrap_bbhx>(m, "OrbitsWrapCPU_bbhx")
#endif

    // Bind the constructor
    .def(py::init<double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(),
         py::arg("dt"), py::arg("N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    // .def(py::init<double, double, int, double, double, int, array_type<double>, array_type<double>, array_type<double>, array_type<int>, array_type<int>, array_type<int>, double>(), 
    //      py::arg("sc_t0"), py::arg("sc_dt"), py::arg("sc_N"), py::arg("ltt_t0"), py::arg("ltt_dt"), py::arg("ltt_N"), py::arg("n_arr"), py::arg("ltt_arr"), py::arg("x_arr"), py::arg("links"), py::arg("sc_r"), py::arg("sc_e"), py::arg("armlength"))
    // Bind member functions
    // .def("get_light_travel_time_wrap", &OrbitsWrap::get_light_travel_time_wrap, "Get the light travel time.")
    // .def("get_pos_wrap", &OrbitsWrap::get_pos_wrap, "Get spacecraft position.")
    // .def("get_normal_unit_vec_wrap", &OrbitsWrap::get_normal_unit_vec_wrap, "Get link normal vector.")
    // You can also expose public data members directly using def_readwrite
    .def_readwrite("orbits", &OrbitsWrap_bbhx::orbits)
    // .def("get_link_ind", &OrbitsWrap::get_link_ind, "Get link index.")
    ;
    
}



PYBIND11_MODULE(cbbhx, m) {
     m.doc() = "BBHx C Backend."; // Optional module docstring

    // Call initialization functions from other files
    response_part(m);
    
    m.def("get_module_path_cpp", &get_module_path_cbbhx, "Returns the file path of the module");

    // Optionally, get the path during module initialization and store it
    // This can cause an AttributeError if not handled carefully, as m.attr("__file__")
    // might not be fully set during the initial call if the module is loaded in
    // a specific way (e.g., via pythonw or as a namespace package).
    try {
        std::string path_at_init = m.attr("__file__").cast<std::string>();
        // std::cout << "Module loaded from: " << path_at_init << std::endl;
        m.attr("module_dir") = py::cast(path_at_init.substr(0, path_at_init.find_last_of("/\\")));
    } catch (py::error_already_set &e) {
         // Handle potential error here, e.g., by logging or setting a default value
        std::cerr << "Could not capture __file__ at init time." << std::endl;
        e.restore(); // Restore exception state for proper Python handling
        PyErr_Clear();
    }
}

