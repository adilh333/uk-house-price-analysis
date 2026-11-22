"""Chart style and figure builders for the reports.

Every figure uses the same palette, left-aligned titles, and a source line.
Functions return the figure so notebooks can display it, and write a PNG when
a path is given.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure

BG = "#F7F4EE"
INK = "#1C2430"
MUTED = "#5E6A75"
TEAL = "#0F6E6B"
TEAL_DARK = "#0B4F4C"
CORAL = "#D4572A"
GOLD = "#C98512"
SLATE = "#C8C0B4"
GRID = "#E4DDD0"
ENGLAND = "#0F6E6B"
WALES = "#C98512"

TYPE_COLORS = {
    "Detached": "#0F6E6B",
    "Semi-detached": "#4E9A90",
    "Terraced": "#C98512",
    "Flat": "#D4572A",
}
TYPE_ORDER = ["Detached", "Semi-detached", "Terraced", "Flat"]

SOURCE_NOTE = "Source: HM Land Registry Price Paid Data  ·  standard residential sales, England and Wales"


def apply_style() -> None:
    """Set a quiet, print-like matplotlib style."""
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": BG,
            "savefig.facecolor": BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.titleweight": "semibold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "axes.axisbelow": True,
            "figure.dpi": 120,
            "savefig.dpi": 160,
        }
    )


def format_gbp(value: float, digits: int = 0) -> str:
    """Compact sterling label, such as ``£295k`` or ``£1.2m``."""
    sign = "-" if value < 0 else ""
    amount = abs(float(value))
    if amount >= 1_000_000:
        return f"{sign}£{amount / 1_000_000:.1f}m"
    if amount >= 10_000:
        return f"{sign}£{amount / 1_000:.{digits}f}k"
    return f"{sign}£{amount:,.0f}"


def _finish(fig: Figure, path: Path | None, note: str = SOURCE_NOTE) -> Figure:
    fig.text(0.0, -0.02, note, fontsize=8, color=MUTED, ha="left", va="top")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight", pad_inches=0.28)
    return fig


def plot_monthly_trend(monthly: pd.DataFrame, path: Path | None = None) -> Figure:
    """Median price as a line, with monthly sales as bars.

    ``monthly`` needs ``month`` (timestamp), ``median_price``, ``sales``, and
    ``provisional``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(11.4, 6.2), layout="constrained")
    volume = ax.twinx()
    colours = [SLATE if not flag else "#E2C2B0" for flag in monthly["provisional"]]
    volume.bar(monthly["month"], monthly["sales"], width=20, color=colours, zorder=1)
    volume.set_ylabel("Sales", color=MUTED)
    volume.tick_params(axis="y", colors=MUTED)
    volume.spines["top"].set_visible(False)
    volume.spines["right"].set_color(GRID)
    volume.grid(False)
    volume.set_ylim(0, monthly["sales"].max() * 3.1)

    complete = monthly.loc[~monthly["provisional"]]
    provisional = monthly.loc[monthly["provisional"]]
    ax.plot(
        complete["month"],
        complete["median_price"],
        color=TEAL_DARK,
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        zorder=3,
    )
    if not provisional.empty:
        bridge = pd.concat([complete.tail(1), provisional], ignore_index=True)
        ax.plot(
            bridge["month"],
            bridge["median_price"],
            color=CORAL,
            linewidth=2.2,
            linestyle="--",
            marker="o",
            markersize=3.5,
            zorder=4,
        )
        start = provisional["month"].min() - pd.Timedelta(days=15)
        ax.axvspan(start, monthly["month"].max() + pd.Timedelta(days=20), color="#F3E2D8", alpha=0.45, zorder=0)

    ax.set_ylim(monthly["median_price"].min() * 0.9, monthly["median_price"].max() * 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    ax.set_ylabel("Median price")
    ax.grid(axis="y")
    ax.set_title("Prices sat near £290k, until the March 2025 deadline", loc="left", pad=14)
    ax.text(
        0.0,
        1.02,
        "Line: monthly median. Bars: sales. The shaded months are still incomplete in the 28 October 2026 file.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    return _finish(fig, path)


def plot_volume_index(indexed: pd.DataFrame, path: Path | None = None) -> Figure:
    """England and Wales sales, each indexed to its own 2024 monthly average.

    ``indexed`` needs ``month``, ``nation``, and ``index``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(11.4, 6.0), layout="constrained")
    colours = {"England": ENGLAND, "Wales": WALES}
    for nation, part in indexed.groupby("nation", sort=False):
        ax.plot(
            part["month"],
            part["index"],
            color=colours.get(str(nation), INK),
            linewidth=2.3,
            label=str(nation),
        )
    deadline = pd.Timestamp("2025-03-01")
    ax.axvline(deadline, color=CORAL, linewidth=1.0, linestyle="--")
    ymax = float(indexed["index"].max())
    ax.text(
        deadline + pd.Timedelta(days=12),
        ymax * 0.96,
        "March 2025\nSDLT threshold ends",
        color=CORAL,
        fontsize=8.5,
        va="top",
    )
    ax.axhline(100, color=SLATE, linewidth=0.8)
    ax.set_ylabel("Sales index (2024 monthly average = 100)")
    ax.set_title("The March 2025 rush was an English stamp-duty effect", loc="left", pad=14)
    ax.text(
        0.0,
        1.02,
        "Wales, which charges Land Transaction Tax rather than SDLT, did not see the same spike or April drop.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y")
    return _finish(fig, path)


def plot_regional_medians(stats: pd.DataFrame, overall_median: float, path: Path | None = None) -> Figure:
    """Horizontal bars of regional median price, highest at the top."""
    apply_style()
    ordered = stats.sort_values("median_price", ascending=True)
    fig, ax = plt.subplots(figsize=(10.6, 6.2), layout="constrained")
    colours = [CORAL if region == "London" else TEAL for region in ordered["region"]]
    ax.barh(ordered["region"], ordered["median_price"], color=colours, height=0.72)
    ax.axvline(overall_median, color=INK, linewidth=0.9, linestyle="--")
    for median, region in zip(ordered["median_price"], ordered["region"]):
        ax.text(
            median + 4000,
            region,
            format_gbp(median),
            va="center",
            ha="left",
            fontsize=9,
            color=INK,
        )
    ax.set_xlim(0, ordered["median_price"].max() * 1.18)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    ax.grid(axis="x")
    ax.set_xlabel("Median price")
    ax.set_title("London's median is about three times the North East's", loc="left", pad=14)
    ax.text(
        0.0,
        1.02,
        f"Dashed line: England and Wales median ({format_gbp(overall_median)}), excluding incomplete months.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    return _finish(fig, path)


def plot_local_authorities(highest: pd.DataFrame, lowest: pd.DataFrame, path: Path | None = None) -> Figure:
    """Highest and lowest local-authority median prices."""
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 6.4), layout="constrained")
    panels = [
        (axes[0], lowest.sort_values("median_price", ascending=True), TEAL, "Lowest medians"),
        (axes[1], highest.sort_values("median_price", ascending=True), CORAL, "Highest medians"),
    ]
    for ax, frame, colour, heading in panels:
        ax.barh(frame["local_authority"], frame["median_price"], color=colour, height=0.72)
        for median, name in zip(frame["median_price"], frame["local_authority"]):
            ax.text(median, name, f"  {format_gbp(median)}", va="center", ha="left", fontsize=8.5, color=INK)
        ax.set_xlim(0, frame["median_price"].max() * 1.28)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
        ax.grid(axis="x")
        ax.set_title(heading, loc="left", fontsize=12)
    fig.suptitle(
        "Burnley at £135k, Kensington and Chelsea at about £1.1m",
        x=0.0,
        ha="left",
        fontsize=14,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.0,
        1.02,
        "Authorities with at least 800 standard sales. Names are the Land Registry district.",
        fontsize=9.5,
        color=MUTED,
        ha="left",
    )
    return _finish(fig, path)


def plot_property_type(stats: pd.DataFrame, path: Path | None = None) -> Figure:
    """Median price and share of sales by property type."""
    apply_style()
    ordered = stats.set_index("property_type").loc[TYPE_ORDER].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), layout="constrained")
    colours = [TYPE_COLORS[name] for name in ordered["property_type"]]
    axes[0].bar(ordered["property_type"], ordered["median_price"], color=colours, width=0.72)
    for label, median in zip(ordered["property_type"], ordered["median_price"]):
        axes[0].text(label, median, format_gbp(median), ha="center", va="bottom", fontsize=9, color=INK)
    axes[0].set_ylim(0, ordered["median_price"].max() * 1.16)
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    axes[0].grid(axis="y")
    axes[0].set_title("Median price", loc="left")
    axes[0].tick_params(axis="x", labelrotation=0)

    axes[1].bar(ordered["property_type"], ordered["share"] * 100, color=colours, width=0.72)
    for label, share in zip(ordered["property_type"], ordered["share"]):
        axes[1].text(label, share * 100, f"{share:.0%}", ha="center", va="bottom", fontsize=9, color=INK)
    axes[1].set_ylim(0, max(ordered["share"] * 100) * 1.2)
    axes[1].set_ylabel("Share of sales")
    axes[1].grid(axis="y")
    axes[1].set_title("Share of sales", loc="left")
    fig.suptitle(
        "Detached homes cost the most; semis are the most common",
        x=0.0,
        ha="left",
        fontsize=14,
        fontweight="semibold",
    )
    return _finish(fig, path)


def plot_new_build(stats: pd.DataFrame, path: Path | None = None) -> Figure:
    """Median price for new and existing homes, by property type.

    ``stats`` needs ``property_type``, ``new_build``, ``median_price``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 5.8), layout="constrained")
    width = 0.36
    positions = np.arange(len(TYPE_ORDER))
    for shift, label, colour in ((-width / 2, "Existing", TEAL), (width / 2, "New build", GOLD)):
        part = stats.loc[stats["new_build"].eq(label)].set_index("property_type").loc[TYPE_ORDER]
        ax.bar(positions + shift, part["median_price"], width=width, color=colour, label=label)
    ax.set_xticks(positions, TYPE_ORDER)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    ax.grid(axis="y")
    ax.legend(frameon=False)
    ax.set_ylabel("Median price")
    ax.set_title("New flats carry a large premium; new detached homes do not", loc="left", pad=14)
    ax.text(
        0.0,
        1.02,
        "Unadjusted medians. New flats are higher in every region; the detached gap is small and mixed.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=MUTED,
        ha="left",
        va="bottom",
    )
    return _finish(fig, path)


def plot_tenure(shares: pd.DataFrame, path: Path | None = None) -> Figure:
    """Stacked share of freehold and leasehold by property type.

    ``shares`` needs ``property_type``, ``tenure``, ``share``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10.2, 5.6), layout="constrained")
    freehold = (
        shares.loc[shares["tenure"].eq("Freehold")].set_index("property_type").loc[TYPE_ORDER, "share"].to_numpy()
    )
    leasehold = (
        shares.loc[shares["tenure"].eq("Leasehold")].set_index("property_type").loc[TYPE_ORDER, "share"].to_numpy()
    )
    ax.bar(TYPE_ORDER, freehold, color=TEAL, label="Freehold", width=0.66)
    ax.bar(TYPE_ORDER, leasehold, bottom=freehold, color=CORAL, label="Leasehold", width=0.66)
    for index, (free, lease) in enumerate(zip(freehold, leasehold)):
        if free > 0.08:
            ax.text(index, free / 2, f"{free:.0%}", ha="center", va="center", color="white", fontsize=10)
        if lease > 0.08:
            ax.text(index, free + lease / 2, f"{lease:.0%}", ha="center", va="center", color="white", fontsize=10)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: f"{v:.0%}"))
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.set_title("Flats are leasehold; houses are almost all freehold", loc="left", pad=12)
    ax.grid(axis="y")
    return _finish(fig, path)


def plot_region_type_heatmap(pivot: pd.DataFrame, path: Path | None = None) -> Figure:
    """Median price by region and property type. ``pivot`` is in pounds."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10.4, 6.4), layout="constrained")
    cmap = LinearSegmentedColormap.from_list("price", ["#F8EFD8", "#E7B15A", "#3E8F88", "#14706C"])
    display = pivot.loc[:, TYPE_ORDER] / 1000
    sns.heatmap(
        display,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".0f",
        annot_kws={"fontsize": 8.5, "color": INK},
        linewidths=1.2,
        linecolor=BG,
        cbar_kws={"label": "Median price (£000)", "shrink": 0.8},
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=0)
    ax.tick_params(axis="y", labelrotation=0)
    ax.set_title("A detached home outprices a flat in every region", loc="left", pad=12)
    return _finish(fig, path)


def plot_price_distribution(frame: pd.DataFrame, path: Path | None = None) -> Figure:
    """Overlapping log-price histograms by property type, on a sample if needed."""
    apply_style()
    fig, ax = plt.subplots(figsize=(10.6, 5.6), layout="constrained")
    sample = frame if len(frame) <= 250_000 else frame.sample(n=250_000, random_state=42)
    bins = np.linspace(np.log10(10_000), np.log10(2_000_000), 48)
    for property_type in TYPE_ORDER:
        values = sample.loc[sample["property_type"].eq(property_type), "price"]
        ax.hist(
            np.log10(values),
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=TYPE_COLORS[property_type],
            label=property_type,
        )
    ticks = [100_000, 200_000, 300_000, 500_000, 1_000_000]
    ax.set_xticks(np.log10(ticks), [format_gbp(tick) for tick in ticks])
    ax.set_ylabel("Density")
    ax.set_xlabel("Sale price")
    ax.legend(frameon=False)
    ax.grid(axis="y")
    ax.set_title("The four property types overlap across a wide price range", loc="left", pad=12)
    return _finish(fig, path)


def plot_model_comparison(
    metrics: pd.DataFrame,
    path: Path | None = None,
    title: str = "Holdout error by model",
) -> Figure:
    """Grouped bars of MAE and median absolute error.

    ``metrics`` needs ``model``, ``mae``, ``median_ae``.
    """
    apply_style()
    fig, ax = plt.subplots(figsize=(10.2, 5.6), layout="constrained")
    positions = np.arange(len(metrics))
    width = 0.36
    ax.bar(positions - width / 2, metrics["mae"], width=width, color=TEAL, label="MAE")
    ax.bar(positions + width / 2, metrics["median_ae"], width=width, color=GOLD, label="Median absolute error")
    for index, row in metrics.reset_index(drop=True).iterrows():
        ax.text(index - width / 2, row["mae"], format_gbp(row["mae"]), ha="center", va="bottom", fontsize=8.5)
        ax.text(
            index + width / 2,
            row["median_ae"],
            format_gbp(row["median_ae"]),
            ha="center",
            va="bottom",
            fontsize=8.5,
        )
    ax.set_xticks(positions, metrics["model"])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    ax.set_ylim(0, metrics["mae"].max() * 1.2)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylabel("Error on 2026 holdout (£)")
    ax.set_title(title, loc="left", pad=12)
    return _finish(fig, path, note="Holdout: standard sales from January to July 2026. Errors are in pounds, not log price.")


def plot_pred_vs_actual(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    path: Path | None = None,
    title: str = "Predicted price against the recorded sale price",
) -> Figure:
    """Hexbin of predicted against actual price, capped for display."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 6.6), layout="constrained")
    cap = 1_250_000
    cmap = LinearSegmentedColormap.from_list("density", ["#F7F4EE", "#7FB9B2", "#0F6E6B", "#1C2430"])
    hexbin = ax.hexbin(
        y_true,
        y_pred,
        gridsize=45,
        cmap=cmap,
        bins="log",
        mincnt=1,
        extent=(0, cap, 0, cap),
    )
    ax.plot([0, cap], [0, cap], color=CORAL, linewidth=1.1, linestyle="--")
    ax.set_xlim(0, cap)
    ax.set_ylim(0, cap)
    ax.set_aspect("equal")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    ax.set_xlabel("Actual price")
    ax.set_ylabel("Predicted price")
    ax.set_title(title, loc="left", pad=12)
    fig.colorbar(hexbin, ax=ax, shrink=0.78, label="Sales (log scale)")
    return _finish(
        fig,
        path,
        note="Display capped at £1.25m so the mass of sales is visible. Metrics use every holdout sale.",
    )


def plot_feature_importance(
    importance: pd.DataFrame,
    path: Path | None = None,
    title: str = "Permutation importance on the 2026 holdout",
) -> Figure:
    """Permutation importance as the increase in pound MAE."""
    apply_style()
    ordered = importance.sort_values("mae_increase", ascending=True)
    fig, ax = plt.subplots(figsize=(10.0, 5.6), layout="constrained")
    ax.barh(ordered["label"], ordered["mae_increase"], color=TEAL, height=0.68, xerr=ordered["mae_increase_std"],
            error_kw={"ecolor": MUTED, "capsize": 2, "linewidth": 0.8})
    for label, value in zip(ordered["label"], ordered["mae_increase"]):
        ax.text(value, label, f"  {format_gbp(value)}", va="center", ha="left", fontsize=9, color=INK)
    low = min(0.0, float(ordered["mae_increase"].min()))
    low = low * 1.25 if low < 0 else 0.0
    ax.set_xlim(low, float(ordered["mae_increase"].max()) * 1.25)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _p: format_gbp(v)))
    ax.grid(axis="x")
    ax.set_xlabel("Increase in MAE when the feature is shuffled")
    ax.set_title(title, loc="left", pad=12)
    return _finish(
        fig,
        path,
        note="Permutation importance on a 12,000-sale sample of the 2026 holdout. Error bars are the standard deviation across repeats.",
    )


def plot_error_by_region(
    stats: pd.DataFrame,
    path: Path | None = None,
    title: str = "Relative error by region",
) -> Figure:
    """Median absolute error as a share of each region's median price.

    ``stats`` needs ``region``, ``median_ae``, ``median_price``.
    """
    apply_style()
    ordered = stats.sort_values("relative_ae", ascending=True)
    fig, ax = plt.subplots(figsize=(10.4, 6.0), layout="constrained")
    colours = [CORAL if region == "London" else TEAL for region in ordered["region"]]
    ax.barh(ordered["region"], ordered["relative_ae"] * 100, color=colours, height=0.72)
    for share, region in zip(ordered["relative_ae"], ordered["region"]):
        ax.text(share * 100 + 0.15, region, f"{share:.0%}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, ordered["relative_ae"].max() * 100 * 1.2)
    ax.set_xlabel("Median absolute error as a share of the regional median")
    ax.grid(axis="x")
    ax.set_title(title, loc="left", pad=12)
    return _finish(fig, path, note="Gradient boosting model, January–July 2026 holdout.")
