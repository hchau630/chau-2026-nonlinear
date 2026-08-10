import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from packaging.version import Version
from seaborn import categorical

from niarb import viz


def unit_lines(ax):
    return [line for line in ax.lines if line.get_label() == "_nolegend_"]


def line_coordinate_pairs(ax):
    return [
        (
            tuple(np.asarray(line.get_xdata(), dtype=float)),
            tuple(np.asarray(line.get_ydata(), dtype=float)),
        )
        for line in unit_lines(ax)
    ]


def ordinary_data():
    return pd.DataFrame(
        {
            "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
            "category": ["A", "B", "C", "A", "B", "C"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )


# catplot tests are written with help from GPT 5.6 Sol with Max effort
class TestCatplot:
    @pytest.fixture(autouse=True)
    def close_figures(self):
        yield
        plt.close("all")

    def test_connects_units_across_ordinary_categories(self):
        grid = viz.catplot(
            data=ordinary_data(),
            x="category",
            y="value",
            units="unit",
            kind="strip",
            jitter=False,
        )

        assert line_coordinate_pairs(grid.ax) == [
            ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0)),
            ((0.0, 1.0, 2.0), (4.0, 5.0, 6.0)),
        ]

    def test_horizontal_orientation(self):
        grid = viz.catplot(
            data=ordinary_data(),
            x="value",
            y="category",
            units="unit",
            kind="strip",
            jitter=False,
        )

        assert line_coordinate_pairs(grid.ax) == [
            ((1.0, 2.0, 3.0), (0.0, 1.0, 2.0)),
            ((4.0, 5.0, 6.0), (0.0, 1.0, 2.0)),
        ]

    def test_hue_and_dodge_connect_hues_within_base_category(self):
        rows = []
        value = 0
        for category in ["A", "B"]:
            for hue in ["H1", "H2"]:
                for unit in ["u1", "u2"]:
                    value += 1
                    rows.append((unit, category, hue, value))
        data = pd.DataFrame(
            rows,
            columns=["unit", "category", "hue", "value"],
        )

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            hue="hue",
            units="unit",
            kind="strip",
            dodge=True,
            jitter=False,
        )

        lines = unit_lines(grid.ax)
        assert len(lines) == 4

        for line in lines:
            x = np.asarray(line.get_xdata())
            assert len(x) == 2
            assert np.ptp(x) == pytest.approx(0.4)
            assert np.floor(np.mean(x) + 0.5) in {0, 1}

    def test_hue_without_dodge_connects_ordinary_categories(self):
        data = ordinary_data()
        data["hue"] = data["unit"].map({"u1": "H1", "u2": "H2"})

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            hue="hue",
            units="unit",
            kind="strip",
            dodge=False,
            jitter=False,
        )

        assert line_coordinate_pairs(grid.ax) == [
            ((0.0, 1.0, 2.0), (1.0, 2.0, 3.0)),
            ((0.0, 1.0, 2.0), (4.0, 5.0, 6.0)),
        ]

    def test_facets_keep_unit_paths_on_their_own_axes(self):
        data = pd.concat(
            [
                ordinary_data().assign(facet="left"),
                ordinary_data().assign(
                    facet="right",
                    value=lambda frame: frame["value"] + 100,
                ),
            ],
            ignore_index=True,
        )

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            col="facet",
            units="unit",
            kind="strip",
            jitter=False,
        )

        left_lines = line_coordinate_pairs(grid.axes_dict["left"])
        right_lines = line_coordinate_pairs(grid.axes_dict["right"])

        assert len(left_lines) == len(right_lines) == 2
        assert max(y for _, ys in left_lines for y in ys) < 100
        assert min(y for _, ys in right_lines for y in ys) > 100

    def test_explicit_order_survives_large_jitter(self):
        data = ordinary_data()
        np.random.seed(11)

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            order=["C", "A", "B"],
            units="unit",
            kind="strip",
            jitter=2,
        )

        # The x coordinates may cross because jitter is deliberately enormous,
        # but each path must retain C -> A -> B semantic order.
        y_paths = [tuple(line.get_ydata()) for line in unit_lines(grid.ax)]
        assert y_paths == [(3.0, 1.0, 2.0), (6.0, 4.0, 5.0)]

    def test_explicit_hue_order_survives_large_dodged_jitter(self):
        data = pd.DataFrame(
            {
                "unit": ["u1", "u2"] * 3,
                "category": ["A"] * 6,
                "hue": ["H1", "H1", "H2", "H2", "H3", "H3"],
                "value": [1, 4, 2, 5, 3, 6],
            }
        )
        np.random.seed(17)

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            hue="hue",
            hue_order=["H3", "H1", "H2"],
            units="unit",
            kind="strip",
            dodge=True,
            jitter=2,
        )

        y_paths = [tuple(line.get_ydata()) for line in unit_lines(grid.ax)]
        assert y_paths == [(3.0, 1.0, 2.0), (6.0, 4.0, 5.0)]

    @pytest.mark.skipif(
        Version(pd.__version__) >= Version("3.0"),
        reason=(
            "seaborn 0.13.2 is incompatible with pandas >= 3.0 when "
            "comparing tuple-valued hue levels with string categories"
        ),
    )
    def test_tuple_hue_levels_are_handled_as_scalar_levels(self):
        hue_1 = ("left", 1)
        hue_2 = ("right", 2)
        data = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u2", "u2"],
                "category": ["A", "A", "A", "A"],
                "hue": [hue_1, hue_2, hue_1, hue_2],
                "value": [1, 2, 3, 4],
            }
        )

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            hue="hue",
            hue_order=[hue_1, hue_2],
            units="unit",
            kind="strip",
            dodge=True,
            jitter=False,
        )

        assert len(unit_lines(grid.ax)) == 2
        assert [tuple(line.get_ydata()) for line in unit_lines(grid.ax)] == [
            (1.0, 2.0),
            (3.0, 4.0),
        ]

    def test_palette_generated_redundant_hue_uses_ordinary_categories(self):
        with pytest.warns(FutureWarning, match="without assigning"):
            grid = viz.catplot(
                data=ordinary_data(),
                x="category",
                y="value",
                palette="deep",
                units="unit",
                kind="strip",
                dodge=True,
                jitter=False,
            )

        assert len(unit_lines(grid.ax)) == 2
        assert [tuple(line.get_ydata()) for line in unit_lines(grid.ax)] == [
            (1.0, 2.0, 3.0),
            (4.0, 5.0, 6.0),
        ]

    def test_missing_intermediate_level_is_bridged(self):
        data = ordinary_data().query("not (unit == 'u1' and category == 'B')")

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            order=["A", "B", "C"],
            units="unit",
            kind="strip",
            jitter=False,
        )

        assert line_coordinate_pairs(grid.ax) == [
            ((0.0, 2.0), (1.0, 3.0)),
            ((0.0, 1.0, 2.0), (4.0, 5.0, 6.0)),
        ]

    def test_duplicate_ordinary_level_raises(self):
        data = pd.concat(
            [
                ordinary_data(),
                ordinary_data().iloc[[0]],
            ],
            ignore_index=True,
        )

        with pytest.raises(ValueError, match="at most one observation"):
            viz.catplot(
                data=data,
                x="category",
                y="value",
                units="unit",
                kind="strip",
                jitter=False,
            )

    def test_duplicate_hue_level_within_base_category_raises(self):
        data = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1"],
                "category": ["A", "A", "A"],
                "hue": ["H1", "H1", "H2"],
                "value": [1, 2, 3],
            }
        )

        with pytest.raises(ValueError, match="at most one observation"):
            viz.catplot(
                data=data,
                x="category",
                y="value",
                hue="hue",
                units="unit",
                kind="strip",
                dodge=True,
                jitter=False,
            )

    def test_categorical_units_do_not_emit_groupby_future_warning(self):
        data = ordinary_data()
        data["unit"] = pd.Categorical(
            data["unit"],
            categories=["u1", "u2", "unused"],
        )

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            grid = viz.catplot(
                data=data,
                x="category",
                y="value",
                units="unit",
                kind="strip",
                jitter=False,
            )

        assert len(unit_lines(grid.ax)) == 2
        assert not [
            warning
            for warning in caught
            if issubclass(warning.category, FutureWarning)
            and "observed" in str(warning.message)
        ]

    def test_line_kws_are_normalized_and_label_is_reserved(self):
        grid = viz.catplot(
            data=ordinary_data(),
            x="category",
            y="value",
            units="unit",
            kind="strip",
            jitter=False,
            line_kws={
                "c": "red",
                "lw": 2.5,
                "ls": "--",
                "alpha": 0.2,
                "zorder": 7,
                "label": "should-not-appear",
            },
        )

        for line in unit_lines(grid.ax):
            assert line.get_color() == "red"
            assert line.get_linewidth() == pytest.approx(2.5)
            assert line.get_linestyle() == "--"
            assert line.get_alpha() == pytest.approx(0.2)
            assert line.get_zorder() == 7
            assert line.get_label() == "_nolegend_"

    def test_default_line_zorder_tracks_point_zorder(self):
        grid = viz.catplot(
            data=ordinary_data(),
            x="category",
            y="value",
            units="unit",
            kind="strip",
            jitter=False,
            zorder=8,
        )

        assert {line.get_zorder() for line in unit_lines(grid.ax)} == {7}
        assert {collection.get_zorder() for collection in grid.ax.collections} == {8}

    def test_native_log_scale_coordinates_are_in_data_space(self):
        data = pd.DataFrame(
            {
                "unit": ["u1", "u1", "u1", "u2", "u2", "u2"],
                "category": [1, 10, 100, 1, 10, 100],
                "value": [1, 2, 3, 4, 5, 6],
            }
        )

        grid = viz.catplot(
            data=data,
            x="category",
            y="value",
            orient="x",
            native_scale=True,
            log_scale=(True, False),
            units="unit",
            kind="strip",
            jitter=False,
        )

        actual = np.asarray(line_coordinate_pairs(grid.ax), dtype=np.float64)
        expected = np.asarray(
            [
                ((1.0, 10.0, 100.0), (1.0, 2.0, 3.0)),
                ((1.0, 10.0, 100.0), (4.0, 5.0, 6.0)),
            ],
            dtype=np.float64,
        )
        np.testing.assert_allclose(actual, expected)

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"kind": "box"}, "kind='strip'"),
            ({"kind": "strip"}, "non-None units"),
        ],
    )
    def test_rejects_line_kws_when_they_cannot_have_an_effect(self, kwargs, message):
        with pytest.raises(ValueError, match=message):
            viz.catplot(
                data=ordinary_data(),
                x="category",
                y="value",
                line_kws={"color": "red"},
                **kwargs,
            )

    def test_unpatched_calls_are_forwarded_normally(self):
        original = categorical._CategoricalPlotter.plot_strips

        grid = viz.catplot(
            data=ordinary_data(),
            x="category",
            y="value",
            kind="strip",
            jitter=False,
        )

        assert grid is not None
        assert categorical._CategoricalPlotter.plot_strips is original
        assert unit_lines(grid.ax) == []

    def test_patch_is_restored_after_success(self):
        original = categorical._CategoricalPlotter.plot_strips

        viz.catplot(
            data=ordinary_data(),
            x="category",
            y="value",
            units="unit",
            kind="strip",
            jitter=False,
        )

        assert categorical._CategoricalPlotter.plot_strips is original

    def test_patch_is_restored_after_exception(self, monkeypatch):
        original = categorical._CategoricalPlotter.plot_strips
        observed = {}

        def explode(*args, **kwargs):
            observed["was_patched"] = (
                categorical._CategoricalPlotter.plot_strips is not original
            )
            raise RuntimeError("sentinel")

        monkeypatch.setattr(viz.sns, "catplot", explode)

        with pytest.raises(RuntimeError, match="sentinel"):
            viz.catplot(
                data=ordinary_data(),
                x="category",
                y="value",
                units="unit",
                kind="strip",
            )

        assert observed["was_patched"]
        assert categorical._CategoricalPlotter.plot_strips is original

    def test_version_error_occurs_before_patching(self, monkeypatch):
        original = categorical._CategoricalPlotter.plot_strips
        monkeypatch.setattr(viz.sns, "__version__", "0.14.0")

        with pytest.raises(RuntimeError, match="requires seaborn 0.13.2"):
            viz.catplot(
                data=ordinary_data(),
                x="category",
                y="value",
                units="unit",
                kind="strip",
            )

        assert categorical._CategoricalPlotter.plot_strips is original


@pytest.mark.parametrize(
    "min, max, bins, expected",
    [
        (-0.1, 0.8, 4, [-0.3, 0.0, 0.3, 0.6, 0.9]),
        (-0.4, 0.5, 4, [-0.6, -0.3, 0.0, 0.3, 0.6]),
        (-0.8, 0.1, 4, [-0.9, -0.6, -0.3, 0.0, 0.3]),
        (-0.5, 0.5, 4, [-0.5, -0.25, 0.0, 0.25, 0.5]),
        (0.1, 1.1, 4, [0.1, 0.35, 0.6, 0.85, 1.1]),
    ],
)
def test_histogram_bin_edges(min, max, bins, expected):
    out = viz.histogram_bin_edges(min, max, bins)
    np.testing.assert_allclose(out, expected)


@pytest.mark.parametrize("estimator", ["mean", "median"])
@pytest.mark.parametrize("errorbar", ["se", "sd", ("pi", 100)])
def test_sample_df(estimator, errorbar):
    df = pd.DataFrame(
        {
            "x": [1, 2, 3],
            "y": [0.5, 1.0, 1.5],
            "yerr": [0.1, 0.2, 0.3],
            "ylow": [0.4, 0.8, 1.2],
            "yhigh": [0.8, 1.4, 2.0],
        }
    )
    yerr = "yerr" if errorbar in {"se", "sd"} else ("ylow", "yhigh")

    if estimator == "mean" and errorbar == ("pi", 100):
        with pytest.raises(ValueError):
            viz.sample_df(df, yerr=yerr, estimator=estimator, errorbar=errorbar)
        return

    out = viz.sample_df(df, yerr=yerr, estimator=estimator, errorbar=errorbar)

    if errorbar in {"se", "sd"}:
        errorbar = {"se": "sem", "sd": "std"}[errorbar]
        out = out.groupby("x", observed=True, as_index=False)["y"].agg(
            y=estimator, yerr=errorbar
        )
        pd.testing.assert_frame_equal(out, df[["x", "y", "yerr"]])
    else:
        out = out.groupby("x", observed=True, as_index=False)["y"].agg(
            y=estimator, ylow="min", yhigh="max"
        )
        pd.testing.assert_frame_equal(out, df[["x", "y", "ylow", "yhigh"]])


def test_sample_df_index():
    df = pd.DataFrame(
        {
            "x": [1, 2],
            "y": [0.5, 1.0],
            "yerr": [0.1, 0.2],
        }
    )
    out = viz.sample_df(df, errorbar="sd", index="idx")
    expected = pd.DataFrame(
        {
            "idx": [0, 0, 1, 1, 2, 2],
            "x": [1, 2, 1, 2, 1, 2],
            "y": [0.5, 1.0, 0.4, 0.8, 0.6, 1.2],
        }
    )
    pd.testing.assert_frame_equal(out, expected)
