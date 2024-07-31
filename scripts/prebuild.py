import shutil


fps_cu_to_cpp = [
    "PhenomHM",
    "Response",
    "Interpolate",
    "WaveformBuild",
    "Likelihood",
]
fps_pyx = ["phenomhm", "response", "interpolate", "waveformbuild", "likelihood"]

for fp in fps_cu_to_cpp:
    shutil.copy("src/" + fp + ".cu", "src/" + fp + ".cpp")

for fp in fps_pyx:
    shutil.copy("src/" + fp + ".pyx", "src/" + fp + "_cpu.pyx")


# setup version file
with open("README.md", "r") as fh:
    lines = fh.readlines()

for line in lines:
    if line.startswith("Current Version"):
        version_string = line.split("Current Version: ")[1].split("\n")[0]

with open("bbhx/_version.py", "w") as f:
    f.write("__version__ = '{}'".format(version_string))


initial_text_for_constants_file = """
# Collection of citations for modules in bbhx package

# Copyright (C) 2020 Michael L. Katz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


"""

initial_text_for_constants_file2 = """

Constants
*******************************

The module :code:`bbhx.utils.constants` houses the constants used throughout the package.

Constants list:

"""

fp_out_name = "bbhx/utils/constants.py"
fp_out_name2 = "docs/source/user/constants.rst"
fp_in_name = "include/constants.h"

# develop few.utils.constants.py
with open(fp_out_name, "w") as fp_out:
    with open(fp_out_name2, "w") as fp_out2:
        fp_out.write(initial_text_for_constants_file)
        fp_out2.write(initial_text_for_constants_file2)
        with open(fp_in_name, "r") as fp_in:
            lines = fp_in.readlines()
            for line in lines:
                if len(line.split()) == 3:
                    if line.split()[0] == "#define":
                        try:
                            _ = float(line.split()[2])
                            string_out = (
                                line.split()[1] + " = " + line.split()[2] + "\n"
                            )
                            fp_out.write(string_out)
                            fp_out2.write(f"\t* {string_out}")

                        except ValueError as e:
                            continue

# for fp in fps_cu_to_cpp:
#     os.remove("src/" + fp + ".cpp")

# for fp in fps_pyx:
#     os.remove("src/" + fp + "_cpu.pyx")
