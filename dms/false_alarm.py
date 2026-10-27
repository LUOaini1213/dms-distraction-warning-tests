"""False-alarm / detection study on SYNTHETIC natural-looking drives (dms.scenarios.natural).

Each drive carries noise-free ground truth: the gaze location (gt_gaze) and the ACTIVITY (gt_activity: road /
driving / non_driving).  Shoulder checks and junction scanning are labelled off_road by location but are
driving activity - exactly the case a label-based DMS cannot tell apart.  Observed labels add blinks,
flicker, tracking dropouts and whole-glance misclassification.  Two distraction episodes per drive are injected
with known start times.

A distraction alert (acoustic onset at t) is
  justified    if the ground-truth ACTIVITY within [t - 1.0 s, t + 0.2 s] shows a non-driving glance >= 3.0 s
               (lower edge of the Euro NCAP long-distraction band) or >= 8.0 s of non-driving activity in the
               preceding 30 s (VATS 10 s minus the SD-202 [2.0] s tolerance);
  false alarm  otherwise.
"justified-natural" = justified but outside the injected episodes (the glance model's own long-glance tail):
correct by definition, possibly still felt as nuisance.
Episode detection: a distraction alert in [t0, t1 + 1 s].  On-time criteria:
  ADDW       long glance: by t0 + 4.0 s at >= 50 km/h (Part 2 3.1); at 20-50 km/h only mandatory if the glance
             lasts >= 6.0 s (by t0 + 6.5 s, Part 2 3.2).  No VATS requirement.
  Euro NCAP  long glance: by t0 + 4.2 s (declared dT_LGA <= 4 s + [0.2] s); VATS: by the time the episode's own
             cumulative non-driving time reaches 12 s (10 s + [2.0] s), if it does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import regulations as R
from .config import ADDW_MIN, NAIVE, PROPOSED, DmsConfig
from .scenarios import ACT_NON_DRIVING, DATA_ORIGIN, NaturalParams, natural
from .system import onsets, run

JUST_GLANCE_S, JUST_VATS_S = 3.0, 8.0


def reference(activity: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Ground truth: current non-driving glance duration and non-driving time in the last 30 s (seconds)."""
    c = (activity == ACT_NON_DRIVING).astype(np.int32)
    run_len = np.zeros(len(c), dtype=np.int32)
    acc = 0
    for i, x in enumerate(c):
        acc = acc + 1 if x else 0
        run_len[i] = acc
    w = int(round(R.ENCAP_VATS_WINDOW_S / dt))
    cs = np.concatenate(([0], np.cumsum(c)))
    idx = np.arange(1, len(c) + 1)
    return run_len * dt, (cs[idx] - cs[np.maximum(0, idx - w)]) * dt


def sweep_configs() -> list[DmsConfig]:
    P = PROPOSED
    return [P, ADDW_MIN, NAIVE,
            P.with_(name="T_long=3.0", t_long_s=3.0, declared_lga_s=3.1),
            P.with_(name="T_long=3.5", t_long_s=3.5, declared_lga_s=3.6),
            P.with_(name="VATS=8s", vats_threshold_s=8.0, vats_pre_s=5.0),
            P.with_(name="VATS=12s", vats_threshold_s=12.0, vats_pre_s=9.0),
            P.with_(name="reset=1.0s", vats_reset_s=1.0),
            P.with_(name="reset=3.0s", vats_reset_s=3.0),
            P.with_(name="gap=0.05s", gap_tol_s=0.05),
            P.with_(name="gap=0.6s", gap_tol_s=0.6),
            P.with_(name="pre=2.5s", t_pre_s=2.5)]


def make_drives(n_drives: int = 16, minutes: float = 15.0, seed: int = 100, p: NaturalParams | None = None):
    drives = []
    for d in range(n_drives):
        scn, eps = natural(minutes * 60.0, seed + d, p)
        g_gl, g_vats = reference(scn["gt_activity"].to_numpy(dtype=object), PROPOSED.dt)
        drives.append((scn, eps, g_gl, g_vats))
    return drives


def _mandatory(ep, scn: pd.DataFrame, dt: float) -> tuple[float, float]:
    """(ADDW, Euro NCAP) latest acceptable alert time for an injected episode, NaN = not mandatory."""
    i0, i1 = int(round(ep.t0 / dt)), int(round(ep.t1 / dt))
    if ep.kind == "long_glance":
        dur = ep.t1 - ep.t0
        if scn["speed_kph"].iat[i0] >= R.ADDW_HIGH_SPEED_KPH:
            addw = ep.t0 + R.ADDW_TEST_FN_HIGH_S
        else:
            addw = ep.t0 + R.ADDW_TEST_FN_LOW_S if dur >= R.ADDW_T_MAX_LOW_S else float("nan")
        return addw, ep.t0 + R.ENCAP_LGA_RANGE_S[1] + R.ENCAP_LD_TOL_S
    own = np.cumsum(scn["gt_activity"].to_numpy(dtype=object)[i0:i1 + 1] == ACT_NON_DRIVING) * dt
    hit = np.flatnonzero(own >= R.ENCAP_VATS_CUM_S + R.ENCAP_VATS_TOL_S - 1e-9)
    return float("nan"), (float(ep.t0 + hit[0] * dt) if len(hit) else float("nan"))


def study(configs: list[DmsConfig] | None = None, drives=None, latency_s: float = 0.1):
    configs = configs or sweep_configs()
    drives = drives or make_drives()
    ev_rows, ep_rows, summ = [], [], []
    for cfg in configs:
        n_fa = n_just_out = n_pre = n_fss = n_drowsy = 0
        hours = fss_time = 0.0
        cfg_eps = []
        for di, (scn, eps, g_gl, g_vats) in enumerate(drives):
            out = run(scn, cfg, latency_s)
            dt = cfg.dt
            hours += len(scn) * dt / 3600.0
            fss_time += float(out["fss"].to_numpy(dtype=bool).sum()) * dt
            t_on = onsets(out, "acoustic", "distraction")
            src = out["source"].astype(str).to_numpy()
            act = scn["gt_activity"].to_numpy(dtype=object)
            for t in t_on:
                i = min(len(scn) - 1, int(round(t / dt)) - 1)
                lo, hi = max(0, int((t - 1.0) / dt)), min(len(scn) - 1, int((t + 0.2) / dt))
                just = g_gl[lo:hi + 1].max() >= JUST_GLANCE_S or g_vats[lo:hi + 1].max() >= JUST_VATS_S
                in_ep = any(ep.t0 <= t <= ep.t1 + 3.0 for ep in eps)
                n_fa += not just
                n_just_out += just and not in_ep
                ev_rows.append({"config": cfg.name, "drive": di, "t_s": round(t, 3), "source": src[i],
                                "speed_kph": round(float(scn["speed_kph"].iat[i]), 1),
                                "gt_activity_at_alert": act[i], "observed_label_at_alert": scn["gaze"].iat[i],
                                "classification": ("justified-episode" if in_ep and just else
                                                   "justified-natural" if just else "false-alarm"),
                                "gt_glance_max_s": round(float(g_gl[lo:hi + 1].max()), 2),
                                "gt_vats_max_s": round(float(g_vats[lo:hi + 1].max()), 2), "data_origin": DATA_ORIGIN})
            n_pre += len(onsets(out, "visual", "pre_alert"))
            n_fss += len(onsets(out, "fss"))
            n_drowsy += sum(len(onsets(out, "acoustic", s)) for s in ("drowsy", "microsleep", "sleep", "unresponsive"))
            for ep in eps:
                hit = t_on[(t_on >= ep.t0) & (t_on <= ep.t1 + 1.0)]
                m_addw, m_encap = _mandatory(ep, scn, dt)
                t_hit = float(hit[0]) if len(hit) else float("nan")
                r = {"config": cfg.name, "drive": di, "episode": ep.kind, "label": ep.label,
                     "t0_s": round(ep.t0, 2), "dur_s": round(ep.t1 - ep.t0, 2),
                     "speed_kph": round(float(scn["speed_kph"].iat[int(round(ep.t0 / dt))]), 1),
                     "detected": len(hit) > 0, "latency_s": round(t_hit - ep.t0, 3) if len(hit) else float("nan")}
                for name, m in (("addw", m_addw), ("encap", m_encap)):
                    r[f"mandatory_{name}_s"] = round(m - ep.t0, 3) if not np.isnan(m) else float("nan")
                    r[f"on_time_{name}"] = (bool(len(hit) > 0 and t_hit <= m + 1e-9)) if not np.isnan(m) else None
                r["data_origin"] = DATA_ORIGIN
                ep_rows.append(r)
                cfg_eps.append(r)
        e = pd.DataFrame(cfg_eps)
        row = {"config": cfg.name, "hours": round(hours, 2)}
        for kind in ("long_glance", "phone_vats"):
            k = e[e.episode == kind]
            row[f"{kind}_n"] = len(k)
            row[f"{kind}_detect_rate"] = round(float(k.detected.mean()), 3)
            for name in ("addw", "encap"):
                m = k[k[f"on_time_{name}"].notna()]
                row[f"{kind}_on_time_{name}"] = round(float(m[f"on_time_{name}"].astype(bool).mean()), 3) if len(m) else float("nan")
            row[f"{kind}_latency_median_s"] = round(float(k.latency_s.median()), 2)
            row[f"{kind}_latency_p95_s"] = round(float(k.latency_s.quantile(0.95)), 2)
        row.update({"false_alarms": n_fa, "false_alarms_per_h": round(n_fa / hours, 2),
                    "justified_natural_per_h": round(n_just_out / hours, 2),
                    "visual_pre_alerts_per_h": round(n_pre / hours, 2), "fss_activations_per_h": round(n_fss / hours, 1),
                    "fss_time_share": round(fss_time / (hours * 3600.0), 3),
                    "drowsiness_alerts_per_h": round(n_drowsy / hours, 2), "data_origin": DATA_ORIGIN})
        summ.append(row)
    return pd.DataFrame(ev_rows), pd.DataFrame(ep_rows), pd.DataFrame(summ)
