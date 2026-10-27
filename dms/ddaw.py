"""DDAW (EU) 2021/1341 Annex I Part 2: manufacturer validation - event classification and acceptance criteria.

Implemented from the official EEA Norwegian translation (English OJ text not reachable; see regulations.py):
  5.1.2   KSS self-rating about every 5 min; each rating covers the preceding 5 min
  5.1.4   any DDAW warning is a true positive (TP) if the previous OR the next rating is KSS >= 7;
          after a TP the rest of that run is irrelevant
  5.1.5   rating below the threshold (8) followed by a rating >= 8 and no warning in that interval:
          false negative (FN) - unless the run continues one more interval, then that next rating decides:
          >= 8 -> FN, = 7 -> true negative flagged as outlier, < 7 -> run excluded (rating unreliable)
  3.1     every counted participant has >= 1 TP or FN; >= 10 participants; sum TP + FN >= 10
  8.1     effective if (a) mean sensitivity > 40 % or (b) mean - 1.645 * SD > 20 %
          (+5 / +2.5 points if the rating interval > 15 min; -5 / -2.5 points on an open road)
  8.2     results in the learning phase (or the first 30 min, whichever is shorter) are excluded
The SD formula is printed as an image in the source; the text fragments read
  SD = sqrt( sum (sensitivity - mean)^2 / number of participants )  -> population SD (reconstruction, unverified).

The synthetic validation run below only demonstrates this computation pipeline.  Its KSS-to-eyelid model
(DA-21) is invented; the numbers say nothing about real drivers and are NOT a DDAW validation.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import labels as L
from . import regulations as R
from .config import PROPOSED
from .scenarios import DATA_ORIGIN
from .system import DmsSystem, onsets, run


def classify(kss_t: list[float], kss: list[int], warn_t: list[float], threshold: int = R.DDAW_KSS_WARN,
             tp_kss: int = R.DDAW_TP_KSS) -> dict:
    """Classify one run.  kss_t[i] is the time of rating i; interval i = (kss_t[i-1], kss_t[i]]."""
    res = {"tp": 0, "fn": 0, "fp": 0, "tn_outlier": 0, "excluded": False}
    w = np.asarray(warn_t, dtype=float)
    for i in range(1, len(kss)):
        prev, nxt = kss[i - 1], kss[i]
        has_w = bool(((w > kss_t[i - 1]) & (w <= kss_t[i])).any())
        if has_w:
            if prev >= tp_kss or nxt >= tp_kss:
                res["tp"] += 1
                break                                   # 5.1.4: later data irrelevant for this run
            res["fp"] += 1
            continue
        if prev < threshold <= nxt:
            if i + 1 < len(kss):
                after = kss[i + 1]
                if after >= threshold:
                    res["fn"] += 1
                elif after == tp_kss:
                    res["tn_outlier"] += 1
                else:
                    res["excluded"] = True
                    break
            else:
                res["fn"] += 1
    return res


def acceptance(sens: list[float], interval_over_15min: bool = False, open_road: bool = False) -> dict:
    s = np.asarray(sens, dtype=float)
    mean = float(s.mean())
    sd = float(math.sqrt(((s - mean) ** 2).sum() / len(s)))
    lb = mean - R.DDAW_LB_Z * sd
    a_req, b_req = R.DDAW_MEAN_SENS_MIN, R.DDAW_LB_SENS_MIN
    if interval_over_15min:
        a_req, b_req = a_req + R.DDAW_ADJ_LONG_INTERVAL[0], b_req + R.DDAW_ADJ_LONG_INTERVAL[1]
    if open_road:
        a_req, b_req = a_req + R.DDAW_ADJ_OPEN_ROAD[0], b_req + R.DDAW_ADJ_OPEN_ROAD[1]
    return {"n": len(s), "mean_sensitivity": round(mean, 4), "sd": round(sd, 4), "lower_bound": round(lb, 4),
            "req_mean": round(a_req, 4), "req_lb": round(b_req, 4), "crit_a": mean > a_req, "crit_b": lb > b_req,
            "effective": bool(mean > a_req or lb > b_req)}


def _eyelid_run(rng: np.random.Generator, kss_curve: np.ndarray, dt: float, susceptibility: float) -> np.ndarray:
    """DA-21: blinks as a Poisson process (0.3 /s); mean blink length grows with KSS (invented model)."""
    n = len(kss_curve)
    eyelid = np.full(n, 0.1) + 0.03 * rng.standard_normal(n)
    t = 0.0
    T = n * dt
    while t < T:
        t += rng.exponential(1 / 0.3)
        i = int(t / dt)
        if i >= n:
            break
        k = kss_curve[i]
        frac = 0.03 + 0.045 * max(0.0, k - 5.0) ** 1.3          # target closed fraction at this KSS
        mean_d = susceptibility * frac / 0.3
        d = mean_d * math.exp(0.4 * rng.standard_normal() - 0.08)
        eyelid[i:i + max(1, int(round(d / dt)))] = 1.0
    return np.clip(eyelid, 0.0, 1.0)


def synthetic_validation(n_participants: int = 12, minutes: float = 120.0, seed: int = 1341,
                         dt: float = 0.1) -> tuple[pd.DataFrame, dict]:
    """SYNTHETIC demonstration of Part 2 on the PROPOSED drowsiness function (simulated environment)."""
    rng = np.random.default_rng(seed)
    cfg = PROPOSED.with_(dt=dt, name="PROPOSED@10Hz")
    sysm = DmsSystem(cfg)
    n = int(minutes * 60 / dt)
    t = np.arange(n) * dt
    rows = []
    for j in range(n_participants):
        k0, k1, gam = rng.uniform(3, 5), rng.uniform(6.5, 9.5), rng.uniform(0.8, 2.0)
        curve = k0 + (k1 - k0) * (t / t[-1]) ** gam
        eyelid = _eyelid_run(rng, curve, dt, float(math.exp(0.3 * rng.standard_normal())))
        gaze = np.where(eyelid >= cfg.eyelid_closed, L.UNKNOWN, L.ROAD).astype(object)
        scn = pd.DataFrame({"t": t, "speed_kph": 100.0, "gaze": gaze, "eyelid": eyelid, "camera_ok": True,
                            "face_detected": True, "eyes_visible": True, "adas_warning": False, "addw_off": False})
        out = run(scn, cfg, 0.0, sysm)
        warn = onsets(out, "acoustic", "drowsy")
        warn = warn[warn > cfg.perclos_window_s]          # 8.2: exclude the learning phase (60 s window)
        kss_t = list(np.arange(0.0, minutes * 60 + 1e-6, R.DDAW_KSS_INTERVAL_S))
        kss = [int(np.clip(round(curve[min(n - 1, int(tt / dt))] + 0.5 * rng.standard_normal()), 1, 9)) for tt in kss_t]
        c = classify(kss_t, kss, list(warn))
        ev = c["tp"] + c["fn"]
        rows.append({"participant": f"S{j + 1:02d}", "kss_start": kss[0], "kss_end": kss[-1], "max_kss": max(kss),
                     "n_warnings": len(warn), **c, "events": ev,
                     "sensitivity": (c["tp"] / ev) if ev and not c["excluded"] else float("nan"),
                     "data_origin": DATA_ORIGIN})
    df = pd.DataFrame(rows)
    used = df[(df.events > 0) & (~df.excluded)]
    summ = {"participants_total": len(df), "participants_counted": len(used),
            "tp_plus_fn": int(used.events.sum()),
            "sample_ok": bool(len(used) >= R.DDAW_MIN_PARTICIPANTS and used.events.sum() >= R.DDAW_MIN_EVENTS)}
    if len(used):
        summ.update(acceptance(list(used.sensitivity)))
    summ["data_origin"] = DATA_ORIGIN
    return df, summ
