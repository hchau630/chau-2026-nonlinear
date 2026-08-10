import math

import numpy as np
from matplotlib import rcParams, colors

CM = 1 / 2.54  # cm to inch
CM_TO_PT = 28.3465  # cm to pt
AXSIZE = (2.1 * CM, 2.1 * CM)
CAXSIZE = (2.5 * CM, 2.1 * CM)
FIGSIZE = (3.6 * CM, 3.4 * CM)
CFIGSIZE = (4.3 * CM, 3.4 * CM)
ABSRECT = (0.95 * CM, 0.9 * CM, AXSIZE[0], AXSIZE[1])
RECT = (
    0.95 * CM / FIGSIZE[0],
    0.9 * CM / FIGSIZE[1],
    AXSIZE[0] / FIGSIZE[0],
    AXSIZE[1] / FIGSIZE[1],
)
CRECT = (
    0.95 * CM / CFIGSIZE[0],
    0.9 * CM / CFIGSIZE[1],
    CAXSIZE[0] / CFIGSIZE[0],
    CAXSIZE[1] / CFIGSIZE[1],
)
GREY = "#666666"
GRID_WIDTH = rcParams["grid.linewidth"]
GRID_COLOR = rcParams["grid.color"]


def get_sizes(leftscale=1.0, rightscale=1.0, topscale=1.0, bottomscale=1.0, cbar=False):
    figsize = CFIGSIZE if cbar else FIGSIZE
    axsize = CAXSIZE if cbar else AXSIZE
    leftextend = figsize[0] * (leftscale - 1)
    rightextend = figsize[0] * (rightscale - 1)
    topextend = figsize[1] * (topscale - 1)
    bottomextend = figsize[1] * (bottomscale - 1)
    figsize = (
        figsize[0] + leftextend + rightextend,
        figsize[1] + topextend + bottomextend,
    )
    rect = (
        (ABSRECT[0] + leftextend) / figsize[0],
        (ABSRECT[1] + bottomextend) / figsize[1],
        axsize[0] / figsize[0],
        axsize[1] / figsize[1],
    )
    return figsize, rect


def get_cbar_configs(spacing, levels, N_levels, linthresh):
    cnorm = colors.CenteredNorm()
    if spacing == "log":
        loghalfrange = max(abs(levels[0]), abs(levels[1]))
        norm = colors.FuncNorm(
            (lambda x: cnorm(np.log10(x)), lambda x: 10 ** cnorm.inverse(x)),
            vmin=10**-loghalfrange,
            vmax=10**loghalfrange,
        )
        levels = np.logspace(*levels, N_levels)
    elif spacing == "halflog":
        norm = colors.SymLogNorm(
            linthresh, vmin=-(10 ** levels[1]), vmax=10 ** levels[1]
        )
        levels = np.logspace(*levels, N_levels)
    elif spacing == "neghalflog":
        norm = colors.SymLogNorm(
            linthresh, vmin=-(10 ** levels[1]), vmax=10 ** levels[1]
        )
        levels = -np.logspace(*levels, N_levels)[::-1]
    elif spacing == "symlog":
        norm = colors.SymLogNorm(linthresh)
        levels = np.logspace(*levels, N_levels)
        levels = np.r_[-levels[::-1], 0, levels]
    elif spacing == "linear":
        norm = colors.CenteredNorm(vcenter=1)
        levels = np.linspace(*levels, N_levels)
    elif spacing == "halflinear":
        norm = cnorm
        levels = np.linspace(*levels, N_levels)

    if spacing == "symlog":
        halfticks = np.arange(
            math.ceil(math.log10(linthresh)),
            math.ceil(math.log10(levels[-1])) + 1,
            1,
        )
        ticks = np.r_[-(10.0**halfticks), 0, 10.0**halfticks]
    elif spacing in {"log", "halflog"}:
        ticks = 10.0 ** np.arange(
            math.ceil(math.log10(levels[0])),
            math.floor(math.log10(levels[-1])) + 1,
            1,
        )
    elif spacing == "neghalflog":
        ticks = 10.0 ** np.arange(
            math.ceil(math.log10(-levels[-1])),
            math.floor(math.log10(-levels[0])) + 1,
            1,
        )
        ticks = -ticks[::-1]
    else:
        ticks = np.linspace(
            levels[0],
            levels[-1],
            1 + (N_levels - 1) // 2 ** math.ceil(math.log2((N_levels - 1) / 3)),
        )
    return levels, norm, ticks


def set_rcParams():
    rcParams["figure.titlesize"] = "medium"  # default: large
    rcParams["axes.titlesize"] = "medium"  # default: large
    rcParams["font.size"] = 7.25  # default: 10 pts
    rcParams["axes.labelpad"] = 0.0  # default: 4.0 pts
    rcParams["axes.titlepad"] = 4.0  # default: 6.0 pts
