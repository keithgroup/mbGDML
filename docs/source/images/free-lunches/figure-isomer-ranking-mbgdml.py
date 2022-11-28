# MIT License
#
# Copyright (c) 2022, Alex M. Maldonado
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from scipy.special import comb
import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

save_path = "./curse-of-dimensionality"
image_type = "png"

line_width = 2
fill_below = True

# Ensures we execute from script directory (for relative paths).
os.chdir(os.path.dirname(os.path.realpath(__file__)))

# More information: https://matplotlib.org/stable/api/matplotlib_configuration_api.html#default-values-and-styling
font_dirs = ["../../roboto-font"]
font_paths = mpl.font_manager.findSystemFonts(fontpaths=font_dirs, fontext="ttf")
for font_path in font_paths:
    mpl.font_manager.fontManager.addfont(font_path)
rc_params = {
    "figure": {"dpi": 1000},
    "font": {"family": "Roboto", "size": 8, "weight": "normal"},
    "axes": {"edgecolor": "#C2C1C1", "labelweight": "heavy", "labelcolor": "#191919"},
    "xtick": {"color": "#C2C1C1", "labelcolor": "#191919", "labelsize": 7},
    "ytick": {"color": "#C2C1C1", "labelcolor": "#191919", "labelsize": 7},
}
for key, params in rc_params.items():
    plt.rc(key, **params)


# Compute data

step_size = 1
min_size = 3
max_size = 500
system_size = np.arange(min_size, max_size + step_size, step_size)
combs_1body = np.array([comb(size, 1) for size in system_size])
combs_2body = np.array([comb(size, 2) for size in system_size])
combs_3body = np.array([comb(size, 3) for size in system_size])
zeros = np.zeros(system_size.shape)

fig, ax = plt.subplots(1, 1, figsize=(3.5, 3.0), constrained_layout=True)

ax.set_yscale("log")
ax.plot(
    system_size,
    combs_1body,
    label="1-body",
    color="#ffafcc",
    linewidth=line_width,
    zorder=5,
)
ax.plot(
    system_size,
    combs_2body,
    label="2-body",
    color="#a2d2ff",
    linewidth=line_width,
    zorder=3,
)
ax.plot(
    system_size,
    combs_3body,
    label="3-body",
    color="#cdb4db",
    linewidth=line_width,
    zorder=1,
)


if fill_below:
    ax.fill_between(system_size, zeros, combs_1body, color="#FFDAE7", zorder=4)
    ax.fill_between(system_size, zeros, combs_2body, color="#D6EBFF", zorder=2)
    ax.fill_between(system_size, zeros, combs_3body, color="#E3D6EB", zorder=0)

ax.set_xlabel("Number of fragments")
ax.set_xlim(left=0, right=max_size)

ax.set_ylabel("Total combinations")
ax.set_ylim(bottom=1)

ax.legend(frameon=False)

plt.savefig(save_path, dpi=1000)
