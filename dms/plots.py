"""Figures (matplotlib, Agg).  English labels so the figures render on any CI font set; every title says SYNTHETIC.

Colours follow the dataviz reference palette (validated with its validate_palette.js):
  categorical slots 1-2 #2a78d6 / #eb6834 (two series), escalation as a one-hue ordinal ramp
  #86b6ef -> #2a78d6 -> #104281 (visual -> acoustic -> haptic), ink / muted / grid tokens for chrome.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import labels as L  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, BASE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
S1, S2 = "#2a78d6", "#eb6834"
RAMP = {"visual": "#86b6ef", "acoustic": "#2a78d6", "haptic": "#104281"}


def _style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)


def fig_timeline(out: pd.DataFrame, scn: pd.DataFrame, path: Path, t0: float, t1: float, events: list[tuple[float, str]]):
    """Gaze label, single-glance timer, VATS sum and HMI channels for one scenario."""
    m = (out["t"] >= t0) & (out["t"] <= t1)
    o, s = out[m], scn[m.to_numpy()]
    fig, axes = plt.subplots(4, 1, figsize=(10, 7.2), sharex=True, gridspec_kw={"height_ratios": [1.5, 1.1, 1.1, 1.0]})
    fig.patch.set_facecolor(SURFACE)
    order = [L.ROAD, L.MIRROR, L.INSTRUMENT, L.INFOTAINMENT, L.PHONE, L.OFF_ROAD, L.UNKNOWN]
    y = s["gaze"].map({g: i for i, g in enumerate(order)}).to_numpy()
    ax = axes[0]
    ax.step(s["t"], y, where="post", color=INK, linewidth=1.2)
    ax.set_yticks(range(len(order)), order)
    ax.set_ylim(-0.5, len(order) - 0.5)
    ax.invert_yaxis()
    ax.set_ylabel("gaze label (input)")
    _style(ax, None)
    for t, txt in events:
        ax.axvline(t, color=MUTED, linewidth=0.8, linestyle=":")
        ax.text(t + 0.3, 6.35, txt, fontsize=7.5, color=INK2, va="bottom")
    ax = axes[1]
    ax.plot(o["t"], o["glance_s"], color=S1, linewidth=2)
    for v, txt in ((2.0, "visual pre-alert 2.0 s"), (3.3, "alert 3.3 s")):
        ax.axhline(v, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text(t1 - 0.3, v + 0.12, txt, fontsize=7.5, color=INK2, ha="right")
    ax.set_ylabel("single glance (s)")
    ax.set_ylim(0, 7)
    _style(ax)
    ax = axes[2]
    ax.plot(o["t"], o["vats_s"], color=S1, linewidth=2)
    for v, txt in ((7.0, "VATS pre-alert 7 s"), (10.0, "VATS alert 10 s")):
        ax.axhline(v, color=MUTED, linewidth=0.8, linestyle="--")
        ax.text(t1 - 0.3, v + 0.3, txt, fontsize=7.5, color=INK2, ha="right")
    ax.set_ylabel("non-driving time\nin 30 s window (s)")
    ax.set_ylim(0, 14)
    _style(ax)
    ax = axes[3]
    chans = [("visual", 2), ("acoustic", 1), ("haptic", 0)]
    t = o["t"].to_numpy()
    for name, row in chans:
        on = o[name].to_numpy(dtype=bool)
        edges = np.flatnonzero(np.diff(np.concatenate(([0], on.astype(int), [0]))))
        for a, b in zip(edges[::2], edges[1::2]):
            ax.add_patch(plt.Rectangle((t[a], row + 0.15), t[b - 1] - t[a] + (t[1] - t[0]), 0.7, color=RAMP[name], linewidth=0))
    fss = o["fss"].to_numpy(dtype=bool)
    ax.fill_between(t, -0.95, -0.35, where=fss, color=GRID, step="post", linewidth=0)
    ax.set_yticks([2.5, 1.5, 0.5, -0.65], ["visual", "acoustic", "haptic", "ADAS sens."])
    ax.set_ylim(-1.1, 3.1)
    ax.set_xlabel("time (s)")
    _style(ax, None)
    ax.set_xlim(t0, t1)
    fig.suptitle("Fig. 1  PROPOSED DMS on a SYNTHETIC scenario (55 km/h, 0.1 s perception latency):\n"
                 "a 6 s phone glance, then repeated 1.8 s phone glances (VATS; second alert within 60 s starts with haptic)",
                 fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_false_alarm(summ: pd.DataFrame, path: Path):
    d = summ.copy()
    names = d["config"].tolist()[::-1]
    d = d.set_index("config").loc[names]
    yy = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.8), sharey=True, gridspec_kw={"width_ratios": [1.1, 1.2, 1.1]})
    fig.patch.set_facecolor(SURFACE)
    hl = [S1 if n == "PROPOSED" else MUTED for n in names]
    ax = axes[0]
    ax.barh(yy, d["false_alarms_per_h"], color=hl, height=0.6)
    for y, v in zip(yy, d["false_alarms_per_h"]):
        ax.text(v + 0.1, y, f"{v:g}", va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(yy, names)
    ax.set_xlabel("false alarms per hour")
    _style(ax, "x")
    ax = axes[1]
    ax.scatter(d["long_glance_on_time_encap"] * 100, yy + 0.12, color=S1, s=36, label="long glance (by 4.2 s)", zorder=3)
    ax.scatter(d["phone_vats_on_time_encap"] * 100, yy - 0.12, color=S2, s=36, marker="s", label="phone VATS (by 12 s)", zorder=3)
    ax.set_xlim(-5, 105)
    ax.set_xlabel("injected episodes alerted on time (%)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower left")
    _style(ax, "x")
    ax = axes[2]
    v = d["phone_vats_latency_median_s"]
    ax.barh(yy, v.fillna(0), color=hl, height=0.6)
    for y, val in zip(yy, v):
        ax.text((0 if val != val else val) + 0.3, y, "no alert" if val != val else f"{val:.1f}", va="center", fontsize=7.5, color=INK2)
    ax.set_xlabel("median VATS alert latency from episode start (s)")
    _style(ax, "x")
    for lab in axes[0].get_yticklabels():
        if lab.get_text() == "PROPOSED":
            lab.set_fontweight("bold")
            lab.set_color(INK)
    hours = d["hours"].iloc[0]
    fig.suptitle(f"Fig. 2  False alarms vs detection on {hours:g} h of SYNTHETIC natural driving "
                 f"(16 drives, 32 injected episodes per configuration)", fontsize=10, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def fig_nhtsa(per: pd.DataFrame, summ: pd.DataFrame, path: Path):
    variants = list(summ["variant"])
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    spec = [("excess_gt2", "crit_a", "glances > 2.0 s minus allowance\n(15 % of glances, rounded up)", 0.0, "a"),
            ("mean_glance_s", "crit_b", "mean glance duration (s)", 2.0, "b"),
            ("teort_s", "crit_c", "total eyes-off-road time (s)", 12.0, "c")]
    per = per.assign(excess_gt2=per["n_glances_gt2s"] - per["allowed_gt2s"])
    for ax, (col, crit, lab, thr, key) in zip(axes, spec):
        for i, v in enumerate(variants):
            g = per[per.variant == v]
            x = i + rng.uniform(-0.18, 0.18, len(g))
            y = g[col].to_numpy(dtype=float) + (rng.uniform(-0.12, 0.12, len(g)) if key == "a" else 0.0)
            ok = g[crit].to_numpy(dtype=bool)
            ax.scatter(x[ok], y[ok], s=22, color=S1, zorder=3)
            ax.scatter(x[~ok], y[~ok], s=22, facecolors="none", edgecolors=S1, linewidths=1.2, zorder=3)
            n = int(summ.set_index("variant").loc[v, f"n_meet_{key}"])
            ax.text(i, 0.96, f"{n}/24 meet", transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8,
                    color=INK, fontweight="bold" if n >= 21 else "normal")
        ax.axhline(thr, color=MUTED, linestyle="--", linewidth=0.9)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.18 * (hi - lo))                     # headroom for the k/24 labels
        ax.set_xticks(range(len(variants)), [v.replace("_", "\n") for v in variants])
        ax.set_title(f"({key}) {lab}", fontsize=9, color=INK)
        _style(ax)
    fig.suptitle("Fig. 3  NHTSA EGDS criteria, task 'change navigation destination', SYNTHETIC glance model, 24 participants\n"
                 "filled = meets the criterion, hollow = fails; dashed line = limit; a task needs >= 21/24 on all three",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
