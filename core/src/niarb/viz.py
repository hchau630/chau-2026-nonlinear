import functools
import importlib
import inspect
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from itertools import accumulate
from numbers import Number
from unittest.mock import patch

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from matplotlib import rcParams
from matplotlib.axes import Axes
from matplotlib.cbook import normalize_kwargs
from matplotlib.colors import LogNorm
from matplotlib.legend import Legend
from matplotlib.markers import MarkerStyle
from matplotlib.offsetbox import AnchoredText
from matplotlib.text import Text
from pandas import DataFrame
from scipy import stats
from seaborn import FacetGrid
from seaborn._statistics import EstimateAggregator

from niarb import utils

CM = 1 / 2.54  # cm to inch
PAPER_FIGURE_HEIGHT = 3.4 * CM
PAPER_RC_PARAMS = {
    "figure.titlesize": 7.25,
    "font.size": 7.25,  # default: 10 pts
    "axes.labelpad": 0.0,  # default: 4.0 pts
    "axes.titlepad": 4.0,  # default: 6.0 pts
}

# constants specific to custom catplot
REQUIRED_SEABORN_VERSION = "0.13.2"
DEFAULT_LINE_KWS = {
    "color": "0.45",
    "alpha": 0.45,
    "linewidth": 0.8,
}

logger = logging.getLogger(__name__)


def mapped(func, mapping):
    @functools.wraps(func)
    def wrapper(data=None, **kwargs):
        keys = {"x", "y", "hue", "col", "row", "style"}
        data = data.rename(columns=mapping)
        kwargs = {
            k: mapping[v] if (k in keys) and (v in mapping) else v
            for k, v in kwargs.items()
        }
        return func(data, **kwargs)

    return wrapper


def cat_logger(source, kwargs):
    columns = [v for k, v in kwargs.items() if k in {"x", "y", "col", "row", "hue"}]

    class LoggingEstimateAggregator(EstimateAggregator):
        def __call__(self, data, var):
            res = super().__call__(data, var)
            index = data.index[0]
            labels = source.loc[index, columns].to_dict()
            logger.info(str(labels))
            logger.info(f"\n{res}")
            return res

    return LoggingEstimateAggregator


def figplot(
    data: DataFrame,
    func: Callable[[DataFrame], FacetGrid] | str | Sequence[str],
    *,
    layers: Sequence[dict] = (),
    statannot: bool = False,
    statannot_kws: dict | None = None,
    errordim: str | Sequence[str] | None = None,
    grid: str | None = None,
    mapping: dict[str, str] | None = None,
    replace: dict | None = None,
    tight_layout: bool = True,
    xlim: tuple[float | None, float | None] = (None, None),
    ylim: tuple[float | None, float | None] = (None, None),
    title_template: str | None = None,
    title_verbosity: int = 1,
    legend_loc: str | int | None = None,
    legend_title: bool = True,
    legend_kwargs: dict | None = None,
    xscale: str = "linear",
    yscale: str = "linear",
    refline: dict | None = None,
    tick_params: dict | None = None,
    despine: dict | None = None,
    spine_kws: dict[str, dict] | None = None,
    height: Number | str = 5,
    rc_params: dict | str | None = None,
    **kwargs,
) -> FacetGrid:
    FUNCS = {
        "relplot": relplot,
        "lmplot": lmplot,
        "displot": displot,
        "heatplot": heatplot,
        "jointplot": jointplot,
        "catplot": catplot,
    }
    AX_FUNCS = {
        "heatmap": heatmap,
        "histplot": histplot,
    }

    if mapping is None:
        mapping = {}

    if isinstance(func, str):
        func = FUNCS[func] if func in FUNCS else getattr(sns, func)
    elif isinstance(func, Sequence):
        if len(func) != 2:
            raise ValueError("func must be a sequence of length 2.")
        func = getattr(importlib.import_module(func[0]), func[1])

    if isinstance(height, str):
        if height != "paper":
            raise ValueError(f"height must be 'paper' or a number, but {height=}.")
        height = PAPER_FIGURE_HEIGHT

    if isinstance(rc_params, str):
        if rc_params != "paper":
            raise ValueError(
                f"rc_params must be 'paper', dict, or None, but {rc_params=}."
            )
        rc_params = PAPER_RC_PARAMS

    # some data cleaning:
    # 1. remove possibly unused categories
    for k, v in data.dtypes.items():
        if isinstance(v, pd.CategoricalDtype):
            data[k] = data[k].cat.remove_unused_categories()

    # 2. ensure unique index
    data = data.reset_index(drop=True)

    if errordim:
        if "y" not in kwargs:
            raise ValueError("errordim requires 'y' to be specified.")

        if isinstance(errordim, str):
            errordim = [errordim]

        estimator = kwargs.get("estimator", "mean")
        by = [
            v
            for k, v in kwargs.items()
            if k in {"x", "hue", "col", "row", "style"} and v is not None
        ]
        logger.debug(f"{by=}, {errordim=}, {estimator=}")

        data = data.groupby(
            by + list(errordim), observed=True, as_index=False, dropna=False
        )[kwargs["y"]].agg(estimator)

    if replace:
        data = data.replace(replace)
        for k in ["col_order", "row_order", "hue_order", "style_order", "order"]:
            if k in kwargs:
                kwargs[k] = [replace.get(v, v) for v in kwargs[k]]

    logger.debug(f"data:\n{data.to_string(max_rows=100)}")

    if func is relplot:
        kwargs["statannot"] = statannot

    with (
        plt.rc_context(rc_params),
        patch.object(sns.categorical, "EstimateAggregator", cat_logger(data, kwargs)),
    ):
        g = mapped(func, mapping)(data, height=height, **kwargs)

        if layers and not isinstance(g, FacetGrid):
            raise TypeError(
                f"`layers` require `func` to produce a FacetGrid, but got {type(g)}."
            )

        for layer in layers:
            if not isinstance(layer, Mapping) or "func" not in layer:
                raise TypeError(
                    "Each element of `layers` must be a Mapping with key 'func'."
                )

            layer = dict(layer)
            layer_func = layer.pop("func")

            if isinstance(layer_func, str):
                if layer_func in AX_FUNCS:
                    layer_func = AX_FUNCS[layer_func]
                else:
                    layer_func = getattr(sns, layer_func)

            if not callable(layer_func):
                raise TypeError(
                    "layer['func'] must resolve to a callable, but got "
                    f"{type(layer_func)}."
                )

            for k, v in layer.items():
                if k in {"x", "y", "hue", "style"} and isinstance(v, str):
                    layer[k] = mapping.get(v, v)
            g.map_dataframe(layer_func, **layer)

        if statannot:
            keys = ["x", "y", "hue", "order", "hue_order"]
            if func == lmplot:
                keys.append("logx")

            func = lmstatplot if func == lmplot else statplot
            statannot_kws = {k: kwargs[k] for k in keys if k in kwargs} | (
                statannot_kws or {}
            )
            statannot_kws = {
                k: mapping[v] if (k in {"x", "y", "hue"}) and (v in mapping) else v
                for k, v in statannot_kws.items()
            }
            logger.debug(f"statannot_kws: {statannot_kws}")
            g.map_dataframe(func, **statannot_kws)

        # more compact subplot titles
        if isinstance(g, sns.FacetGrid):
            row_temp = "{row_var}: {row_name}" if title_verbosity >= 1 else "{row_name}"
            col_temp = "{col_var}: {col_name}" if title_verbosity >= 1 else "{col_name}"
            sep = "\n" if title_verbosity >= 1 else ", "
            if "row" in kwargs and "col" in kwargs:
                temp = f"{row_temp}{sep}{col_temp}"
            else:
                temp = None
            temp = temp if title_template is None else title_template
            g.set_titles(template=temp, row_template=row_temp, col_template=col_temp)

        # set xlim, ylim
        if xlim != (None, None) or ylim != (None, None):
            g.set(xlim=xlim, ylim=ylim)

        # set xscale, yscale
        if xscale != "linear" or yscale != "linear":
            # for some reason combining this with xlim, ylim causes issues in edge cases
            g.set(xscale=xscale, yscale=yscale)

        # add gridlines, with options for adding x = 0 or y = 0 lines.
        if grid in {"xzero", "yzero", "xyzero"}:
            color, linewidth = rcParams["grid.color"], rcParams["grid.linewidth"]
            _x = 0 if grid in {"xzero", "xyzero"} else None
            _y = 0 if grid in {"yzero", "xyzero"} else None
            refline_kwargs = {"linestyle": "-", "color": color, "linewidth": linewidth}
            if func is jointplot:
                refline_kwargs["marginal"] = False
            g.refline(x=_x, y=_y, **refline_kwargs)
        elif grid:
            for ax in g.axes.flat:
                ax.grid(axis=grid)

        if refline:
            g.refline(**refline)

        if tick_params:
            g.tick_params(**tick_params)

        if despine:
            g.despine(**despine)

        if spine_kws:
            axes = [g.ax_marg_x, g.ax_marg_y] if func is jointplot else g.axes.flat
            for ax in axes:
                for k, v in spine_kws.items():
                    ax.spines[k].set(**v)

        # format legend
        if hasattr(g, "legend") and g.legend:
            if legend_loc is not None:
                sns.move_legend(g, legend_loc, **(legend_kwargs or {}))
            if not legend_title:
                g.legend.set_title(None)

        # call tight_layout
        if tight_layout:
            g.tight_layout()

    return g


def jointplot(
    data=None,
    *,
    joint_kind="scatter",
    marginal_kind="hist",
    joint_kws=None,
    marginal_kws=None,
    **kwargs,
):
    joint_func = histplot if joint_kind == "hist" else getattr(sns, f"{joint_kind}plot")
    marginal_func = (
        histplot if marginal_kind == "hist" else getattr(sns, f"{marginal_kind}plot")
    )

    def _marginal_func(
        x, vertical=False, palette=None, hue_order=None, hue_norm=None, **kwargs
    ):
        # need to let seaborn know that we accept these three arguments
        kwargs = kwargs | {
            "palette": palette,
            "hue_order": hue_order,
            "hue_norm": hue_norm,
        }
        if vertical:
            return marginal_func(y=x, **kwargs)
        return marginal_func(x=x, **kwargs)

    marginal_kws = {"legend": False} | marginal_kws

    g = sns.JointGrid(data, **kwargs)
    g.plot_joint(joint_func, **joint_kws)
    g.plot_marginals(_marginal_func, **marginal_kws)

    return g


def heatplot(
    data=None,
    *,
    hue=None,
    vmin=None,
    vmax=None,
    share_hue=True,
    hue_scale=None,
    estimator=None,
    cbar_kws=None,
    **kwargs,
):
    keys = set(inspect.signature(FacetGrid).parameters.keys()) - {"hue"}
    facet_kws = {k: kwargs.pop(k) for k in set(kwargs.keys()) if k in keys}
    facet_kws |= kwargs.pop("facet_kws", {})

    if share_hue:
        if vmin is not None or vmax is not None:
            raise ValueError("Cannot specify vmin or vmax when share_hue is True.")

        if estimator is None:
            vmin, vmax = data[hue].min(), data[hue].max()
        else:
            by = [
                v
                for k, v in (kwargs | facet_kws).items()
                if k in {"x", "y", "col", "row"} and v is not None
            ]
            agg = data.groupby(by, observed=True)[hue].agg(estimator)
            vmin, vmax = agg.min(), agg.max()

    if hue_scale == "log":
        norm = LogNorm(vmin=vmin, vmax=vmax)
        vmin, vmax = None, None
    elif hue_scale is None:
        norm = None
    else:
        raise ValueError(f"Invalid hue_scale: {hue_scale}.")

    cbar_kws = {"label": hue} | (cbar_kws or {})

    g = sns.FacetGrid(data, **facet_kws)
    g.map_dataframe(
        heatmap,
        hue=hue,
        vmin=vmin,
        vmax=vmax,
        estimator=estimator,
        norm=norm,
        cbar_kws=cbar_kws,
        **kwargs,
    )

    return g


def heatmap(
    data: DataFrame = None,
    *,
    x=None,
    y=None,
    hue=None,
    color=None,
    estimator=None,
    **kwargs,
):
    if estimator is None:
        data = data.pivot(index=y, columns=x, values=hue)
    else:
        data = data.pivot_table(index=y, columns=x, values=hue, aggfunc=estimator)
    return sns.heatmap(data, **kwargs)


def relplot(data=None, *, x=None, y=None, statannot=False, **kwargs):
    if utils.is_interval_dtype(data[x].dtype):
        data = data.copy()
        data[x] = utils.get_interval_mid(data[x])
    logger.debug(f"data:\n{data}")
    logger.debug(f"data memory usage:\n{data.memory_usage()}")

    # for some reason seaborn is very memory-inefficient, so manually do groupby
    # if errorbar is "se", "sd", or None. This is important for plotting large
    # dataframes such as when plotting weights. Also avoid doing this if
    # statannot is True, since we need to pass the raw data to statplot.
    if (
        "errorbar" in kwargs
        and statannot is False
        and any(kwargs["errorbar"] == k for k in ("se", "sd", None))
    ):
        errorbar = kwargs["errorbar"]
        by = [x] + [
            v
            for k, v in kwargs.items()
            if k in {"x", "hue", "col", "row", "style"} and v is not None
        ]
        agg = {y: "mean"}
        if errorbar == "se":
            agg[f"{y}_{errorbar}"] = "sem"
        elif errorbar == "sd":
            agg[f"{y}_{errorbar}"] = "std"
        data = data.groupby(by, observed=True, as_index=False)[y].agg(**agg)
        if errorbar:
            data = sample_df(data, errorbar=errorbar, y=y, yerr=f"{y}_{errorbar}")
    logger.debug(f"grouped data:\n{data}")

    return sns.relplot(data=data, x=x, y=y, **kwargs)


def lmplot(data=None, *, x=None, **kwargs):
    if utils.is_interval_dtype(data[x].dtype):
        data = data.copy()
        data[x] = utils.get_interval_mid(data[x])
    logger.debug(f"data:\n{data}")
    logger.debug(f"data memory usage:\n{data.memory_usage()}")

    return sns.lmplot(data=data, x=x, **kwargs)


def lmstatplot(
    data=None,
    *,
    x=None,
    y=None,
    logx=False,
    loc="upper right",
    alpha=0.5,
    verbosity=1,
    color=None,
    label=None,
    marker=None,
    method=None,
    n_resamples=9999,
    rng=None,
    **kwargs,
):
    if logx:
        data = data.copy()
        data[x] = np.log10(data[x])

    fit = sm.OLS(data[y], sm.add_constant(data[x])).fit()
    pvalue = fit.pvalues.loc[x]

    if method == "permutation":
        method_ = stats.PermutationMethod(n_resamples=n_resamples, rng=rng)
        pvalue = stats.pearsonr(data[x], data[y], method=method_).pvalue
    elif method == "bootstrap":
        resamples = stats.bootstrap(
            (data[x], data[y]),
            lambda x, y, axis: stats.pearsonr(x, y, axis=axis).statistic,
            paired=True,
            n_resamples=n_resamples,
            method="percentile",  # should agree with seaborn CI exactly
            rng=rng,
        ).bootstrap_distribution
        resamples = resamples[np.isfinite(resamples)]
        pvalue = min(1.0, 2 * min((resamples <= 0).mean(), (resamples >= 0).mean()))
    elif method is not None:
        raise ValueError(
            f"method must be 'permutation', 'bootstrap', or None, but got {method}."
        )

    text = [
        rf"Slope: {fit.params.loc[x]:.2g}$\pm${fit.bse.loc[x]:.2g}",
        rf"Intercept: {fit.params.loc['const']:.2g}$\pm${fit.bse.loc['const']:.2g}",
        f"$R^2$: {fit.rsquared:.2g}, P-value: {pvalue:.2g}",
    ]

    if verbosity == 0:
        text = text[:-1] + [f"$R^2$: {fit.rsquared:.2g}", f"p = {pvalue:.2g}"]

    if verbosity <= 0:
        spec = plt.gca().get_subplotspec()
        if spec is not None:
            _, ncols, start, _ = spec.get_geometry()
            row, col = divmod(start, ncols)
            text = [f"Subplot ({row}, {col}):"] + text

        info, text = (text[:-1], text[-1:]) if verbosity == 0 else (text, [])
        logger.info("\n".join(info))

    text = AnchoredText("\n".join(text), loc, **kwargs)
    text.patch.set_alpha(alpha)
    plt.gca().add_artist(text)
    return text


def displot(data=None, *, kind="hist", legend=True, **kwargs):
    kind = histplot if kind == "hist" else getattr(sns, f"{kind}plot")

    hue_params = {"hue", "hue_order", "palette"}
    keys = set(inspect.signature(FacetGrid).parameters.keys()) - hue_params
    facet_kws = {k: kwargs.pop(k) for k in set(kwargs.keys()) if k in keys}
    facet_kws |= kwargs.pop("facet_kws", {})

    g = sns.FacetGrid(data, **facet_kws)
    g.map_dataframe(kind, **kwargs)
    if legend:
        g.add_legend()

    return g


def histplot(
    data=None,
    *,
    x=None,
    y=None,
    hue=None,
    hue_order=None,
    color=None,
    label=None,
    palette=None,
    bins="auto",
    estimator=None,
    outliers=(0, 1),
    **kwargs,
):
    # sensible handling of data consisting of a single unique value
    if isinstance(x, str) and y is None and data[x].nunique() == 1:
        return plt.vlines(
            data[x].unique().item(), 0, len(data), color=color, label=label
        )

    # remove outlier values
    if outliers != (0, 1):
        lq, uq = outliers
        if x is not None:
            values = x if data is None else data[x]
            mask = (values >= values.quantile(lq)) & (values <= values.quantile(uq))
            if data is None:
                x = x.loc[mask]
            else:
                data = data.loc[mask]
        if y is not None:
            values = y if data is None else data[y]
            mask = (values >= values.quantile(lq)) & (values <= values.quantile(uq))
            if data is None:
                x = x.loc[mask]
            else:
                data = data.loc[mask]

    if (
        isinstance(x, str)
        and y is None
        and isinstance(bins, Sequence)
        and not isinstance(bins, str)
    ):
        if len(bins) < 1:
            raise ValueError(
                "If bins is a sequence, it must have at least one element, but "
                f"{len(bins)=}."
            )

        if isinstance(bins[0], str):
            name, *args = bins
            if name == "zero":
                bins = histogram_bin_edges(data[x].min(), data[x].max(), *args)
            elif name == "circular":
                if len(args) < 3:
                    raise ValueError(
                        "For 'circular' mode, must provide at least 3 bins."
                    )
                data, bins = data.copy(), args
                data.loc[data[x] < bins[1], x] += bins[-1] - bins[0]
                bins[-1] += bins[1] - bins[0]
                bins = bins[1:]
                logger.debug(f"Modified bins: {bins}. Modified x:\n{data[x]}")
            else:
                raise ValueError(f"Invalid binning scheme: {name}.")

    ax = sns.histplot(
        data,
        x=x,
        y=y,
        hue=hue,
        hue_order=hue_order,
        color=color,
        palette=palette,
        label=label,
        bins=bins,
        **kwargs,
    )
    if estimator and (x is None or y is None):
        axline = ax.axvline if x is not None else ax.axhline
        x = x if x is not None else y
        if data is None:
            data = pd.DataFrame({"x": x, "hue": hue})
            x, hue = "x", "hue"
        if hue:
            estimates = data.groupby(hue, observed=True)[x].agg(estimator)
            if hue_order:
                if set(hue_order) != set(estimates.index):
                    raise NotImplementedError()
                estimates = [estimates.loc[h] for h in hue_order]
                logger.debug(f"{hue_order=}, {estimates=}")
        else:
            estimates = [data[x].agg(estimator)]
            logger.debug(f"{estimates=}")
        palette = sns.color_palette(palette)
        for c, v in zip(palette, estimates):
            axline(v, ls="--", c=c, lw=rcParams["grid.linewidth"])
    return ax


def statplot(
    data: DataFrame | None = None,
    *,
    x: str | None = None,
    y: str | None = None,
    order: Sequence[str] | None = None,
    hue: str | None = None,
    hue_order: Sequence[str] | None = None,
    kind: str = "nsamp",
    test: Callable | str | None = None,
    test_kws: dict | None = None,
    alphas: Sequence[float] = (0.05, 0.01, 0.001),
    ax: Axes | None = None,
    ha: str = "center",
    va: str = "center",
    **kwargs,
) -> Text:
    if x is None or y is None:
        raise ValueError("x and y cannot be None.")

    if kind not in {"nsamp", "1samp"}:
        raise ValueError(f"'kind' must be either 'nsamp' or '1samp', but got {kind=}.")

    if kind == "1samp" and hue is not None:
        raise ValueError(f"'hue' is not supported for '1samp' tests, but got {hue=}.")

    if test is None:
        test = {"nsamp": "f_oneway", "1samp": "ttest_1samp"}[kind]

    if test_kws is None:
        test_kws = {}

    if isinstance(test, str):
        test = getattr(stats, test)

    if utils.is_interval_dtype(data[x].dtype):
        data = data.copy()
        data[x] = utils.get_interval_mid(data[x])

    logger.debug(f"data:\n{data}")
    xs, dfs = zip(*data.groupby(x, observed=True))
    if order:
        if set(xs) != set(order):
            raise NotImplementedError()
        dfs = [dfs[xs.index(v)] for v in order]
        xs = order
    logger.debug(f"xs:\n{xs}")

    if hue is not None:
        pvalues = []
        for _x, df in zip(xs, dfs):
            hues, samples = zip(*df.groupby(hue, observed=True)[y])
            if hue_order:
                if set(hues) != set(hue_order):
                    raise NotImplementedError()
                samples = [samples[hues.index(v)] for v in hue_order]
            logger.debug(
                "x:%s, samples:\n%s",
                _x,
                "\n".join(str(s.tolist()) for s in samples),
            )
            pvalues.append(test(*samples, **test_kws).pvalue.item())
    else:
        samples = [df[y] for df in dfs]
        logger.debug("samples:\n%s", "\n".join(str(s.tolist()) for s in samples))
        if kind == "nsamp":
            pvalues = [test(*samples, **test_kws).pvalue.item()]
        else:
            pvalues = [test(s, **test_kws).pvalue.item() for s in samples]

    if len(pvalues) > 1:
        logger.info(
            ", ".join(
                [
                    f"p-values: {dict(zip(xs, pvalues, strict=True))}",
                    f"order: {hue_order}" if hue_order else "",
                ]
            )
        )
    else:
        logger.info(f"p-value: {pvalues[0]}")

    texts = []
    for pvalue in pvalues:
        if np.isnan(pvalue):
            logger.warning("p-value is NaN.")
            text = None
        elif pvalue >= alphas[0]:
            text = None
        elif pvalue >= alphas[1]:
            text = "*"
        elif pvalue >= alphas[2]:
            text = "**"
        else:
            text = "***"
        texts.append(text)

    if not all(isinstance(x, Number) for x in xs):
        xs = range(len(xs))

    if len(pvalues) == 1:
        xs = [sum(xs) / len(xs)]

    if ax is None:
        ax = plt.gca()

    y = ax.get_ylim()[1]
    it = list(zip(xs, texts, strict=True))
    logger.debug(str(it))
    objs = [ax.text(x, y, text, ha=ha, va=va, **kwargs) for x, text in it if text]
    return objs


# catplot and the helper functions _resolve_line_kws, _make_plot_strips are written with
# help from GPT 5.6 Sol with Max effort
def _resolve_line_kws(line_kws):
    """Return normalized Line2D properties without mutating caller input."""
    user_kws = normalize_kwargs(
        {} if line_kws is None else dict(line_kws),
        mpl.lines.Line2D,
    )

    resolved = DEFAULT_LINE_KWS.copy()
    resolved.update(user_kws)

    # Avoid adding one legend entry for every unit.
    resolved["label"] = "_nolegend_"
    return resolved


def _make_plot_strips(line_kws, scatter_legend_artist):
    """Create the seaborn method replacement for one wrapper invocation."""

    # plot_strips is identical to seaborn.categorical._CategoricalPlotter.plot_strips
    # in version 0.13.2 of seaborn except for additional code for plotting unit lines
    def plot_strips(
        self,
        jitter,
        dodge,
        color,
        plot_kws,
    ):

        width = 0.8 * self._native_width
        offsets = self._nested_offsets(width, dodge)

        if jitter is True:
            jlim = 0.1
        else:
            jlim = float(jitter)
        if "hue" in self.variables and dodge and self._hue_map.levels is not None:
            jlim /= len(self._hue_map.levels)
        jlim *= self._native_width
        jitterer = partial(np.random.uniform, low=-jlim, high=+jlim)

        iter_vars = [self.orient]
        if dodge:
            iter_vars.append("hue")

        ax = self.ax
        dodge_move = jitter_move = 0

        has_units = "units" in self.variables
        unit_line_data = {}

        connect_over_hue = (
            has_units
            and "hue" in self.variables
            and dodge
            and not getattr(self, "_redundant_hue", False)
        )

        if "marker" in plot_kws and not MarkerStyle(plot_kws["marker"]).is_filled():
            plot_kws.pop("edgecolor", None)

        for sub_vars, sub_data in self.iter_data(
            iter_vars,
            from_comp_data=True,
            allow_empty=True,
        ):
            ax = self._get_axes(sub_vars)

            if offsets is not None and (offsets != 0).any():
                dodge_move = offsets[sub_data["hue"].map(self._hue_map.levels.index)]

            jitter_move = jitterer(size=len(sub_data)) if len(sub_data) > 1 else 0

            adjusted_data = sub_data[self.orient] + dodge_move + jitter_move
            sub_data[self.orient] = adjusted_data
            self._invert_scale(ax, sub_data)

            if has_units and not sub_data.empty:
                line_data = sub_data[["x", "y", "units"]].copy()

                if connect_over_hue:
                    # Copy from the Series so tuple-valued hue levels are
                    # treated as scalar levels rather than row sequences.
                    line_data["_connect"] = sub_data["hue"]
                    line_data["_base"] = sub_vars[self.orient]
                    line_data["_order"] = self._hue_map.levels.index(sub_vars["hue"])
                else:
                    # The pre-jitter computed coordinate encodes categorical
                    # order, including an explicit `order`.
                    line_data["_connect"] = sub_vars[self.orient]
                    line_data["_order"] = sub_vars[self.orient]

                unit_line_data.setdefault(ax, []).append(line_data)

            points = ax.scatter(sub_data["x"], sub_data["y"], color=color, **plot_kws)
            if "hue" in self.variables:
                points.set_facecolors(self._hue_map(sub_data["hue"]))

        unit_line_kws = line_kws.copy()
        unit_line_kws.setdefault("zorder", plot_kws.get("zorder", 3) - 1)

        for line_ax, frames in unit_line_data.items():
            point_data = pd.concat(frames, ignore_index=True)

            group_cols = ["units"]
            if connect_over_hue:
                # Hold the ordinary category fixed while connecting hues.
                group_cols.append("_base")

            required = ["x", "y", "units", "_connect", "_order"]
            if connect_over_hue:
                required.append("_base")

            point_data = point_data.dropna(subset=required)

            identity_cols = [*group_cols, "_connect"]
            if point_data.duplicated(identity_cols, keep=False).any():
                mode = (
                    "hue level within an ordinary category"
                    if connect_over_hue
                    else "categorical level"
                )
                raise ValueError(
                    "Cannot draw unambiguous unit lines: each unit must have "
                    f"at most one observation per {mode}."
                )

            for _, path in point_data.groupby(group_cols, sort=False, observed=True):
                # A line requires at least two distinct connected levels.
                if path["_connect"].nunique(dropna=False) < 2:
                    continue

                # Sort by semantic order, not by the potentially crossed
                # jittered coordinates.
                path = path.sort_values("_order", kind="mergesort")

                line_ax.plot(
                    path["x"].to_numpy(), path["y"].to_numpy(), **unit_line_kws
                )

        self._configure_legend(ax, scatter_legend_artist, common_kws=plot_kws)

    return plot_strips


def catplot(*args, line_kws=None, **kwargs):
    """
    Call seaborn.catplot with optional unit lines on a strip plot.

    For ``kind="strip"`` with a non-None ``units`` semantic, observations
    belonging to the same unit are connected using their final displayed
    coordinates.

    When a nonredundant hue is dodged, hue levels are connected within each
    ordinary category. Otherwise, ordinary categories are connected.

    ``line_kws`` accepts Matplotlib Line2D properties.
    """
    kind = kwargs.get("kind", "strip")
    units = kwargs.get("units")

    if kind != "strip" or units is None:
        if line_kws is not None:
            raise ValueError(
                "line_kws requires kind='strip' and a non-None units argument."
            )
        return sns.catplot(*args, **kwargs)

    if sns.__version__ != REQUIRED_SEABORN_VERSION:
        raise RuntimeError(
            "This patch requires seaborn "
            f"{REQUIRED_SEABORN_VERSION}; found {sns.__version__}."
        )

    # Import private seaborn objects only after validating the version.
    from seaborn import categorical
    from seaborn.utils import _scatter_legend_artist

    replacement = _make_plot_strips(_resolve_line_kws(line_kws), _scatter_legend_artist)

    with patch.object(
        categorical._CategoricalPlotter,
        "plot_strips",
        new=replacement,
    ):
        return sns.catplot(*args, **kwargs)


def histogram_bin_edges(min, max, bins):
    """
    Returns equally spaced histogram bin edges where
    0 is one of the edges if min < 0 < max, otherwise
    it just returns np.linspace(min, max, num=bins + 1)
    """
    if min >= max:
        raise ValueError(f"min must be smaller than max, but {min=}, {max=}.")

    if min >= 0 or max <= 0:
        return np.linspace(min, max, num=bins + 1)

    if not isinstance(bins, int) or bins < 2:
        raise ValueError(f"bins must be an integer that is at least 2, but {bins=}.")

    binwidth = (max - min) / bins
    amin = abs(min)
    N_pos = round(max / binwidth)
    N_neg = round(amin / binwidth)
    if math.isclose(amin, binwidth * N_neg) and math.isclose(max, binwidth * N_pos):
        return np.linspace(min, max, num=bins + 1)

    binwidth = (max - min) / (bins - 1)
    N_pos = math.ceil(max / binwidth)
    N_neg = bins - N_pos
    return np.linspace(-binwidth * N_neg, binwidth * N_pos, num=bins + 1)


def remove_legend_subtitles(ax: Axes, nums: Sequence[int], **kwargs) -> Legend:
    handles, labels = ax.get_legend_handles_labels()
    assert len(handles) == len(labels)

    if sum(nums) + len(nums) != len(handles):
        raise ValueError(
            "sum(nums) + len(nums) must equal the number of handles, but "
            f"{sum(nums) + len(nums)=}, {len(handles)=}."
        )

    cumnums = {0} | set(accumulate(n + 1 for n in nums))
    indices = [i for i in range(len(handles)) if i not in cumnums]
    return ax.legend(
        [handles[i] for i in indices], [labels[i] for i in indices], **kwargs
    )


def sample_df(
    df: DataFrame,
    estimator: str = "mean",
    errorbar: str | tuple[str, int] = "se",
    y: str = "y",
    yerr: str | tuple[str, str] = "yerr",
    index: str | None = None,
) -> DataFrame:
    """Generate 'samples' of dataframe

    This can be applied to dataframes with precomputed errorbars
    so that those errorbars can be plotted in seaborn.

    Args:
        df: Dataframe
        estimator (optional): {"mean", "median"}. Estimator for the target variable.
        errorbar (optional): {"se", "sd", ("pi", 100)}. Errorbar type. If ("pi", 100),
          estimator must be "median".
        y (optional): Target variable
        yerr (optional): Errorbar variable. If a tuple, the first element is the
          lower errorbar and the second element is the upper errorbar.
        index (optional): If not None, create a new column with this name containing
          the indices of the samples.

    Returns:
        Dataframe containing 'samples' of the original dataframe

    """
    df0, df1 = df.copy(), df.copy()
    if errorbar in {"se", "sd"}:
        if not isinstance(yerr, str):
            raise ValueError(
                f"yerr must be a string if errorbar is 'se' or 'sd', but {yerr=}."
            )
        scaling = {"se": 3**0.5, "sd": 1}[errorbar]
        df0[y] = df[y] - df[yerr] * scaling
        df1[y] = df[y] + df[yerr] * scaling
    else:
        if not isinstance(yerr, tuple) or len(yerr) != 2:
            raise ValueError(
                f"yerr must be a 2-tuple if errorbar == ('pi', 100), but {yerr=}."
            )
        if estimator != "median":
            raise ValueError(
                "estimator must be 'median' if errorbar == ('pi', 100), but "
                f"{estimator=}."
            )
        df0[y] = df[yerr[0]]
        df1[y] = df[yerr[1]]
        yerr = list(yerr)

    out = pd.concat(dict(enumerate([df, df0, df1]))).drop(columns=yerr)

    if index is not None:
        out = out.reset_index(0, names=index)

    return out.reset_index(drop=True)
