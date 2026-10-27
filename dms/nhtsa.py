"""NHTSA Visual-Manual Driver Distraction Guidelines (78 FR 24818, 2013) - EGDS acceptance on SYNTHETIC glances.

Acceptance (VI.E.14): a testable task should be locked out while driving unless, for at least 21 of the 24
participants (VI.A.4: six per age group 18-24 / 25-39 / 40-54 / 55+; VI.A.5: 12 men, 12 women),
  (a) no more than 15 percent (rounded up) of the glances away from the forward road are longer than 2.0 s,
  (b) the mean glance duration away from the forward road is <= 2.0 s,
  (c) the total eyes-off-road time (TEORT) is <= 12.0 s, performing the task once.
Glance duration includes the transition towards the target (IV.A.10).

Scope warning: Phase 1 of the Guidelines covers VISUAL-MANUAL interfaces only; auditory-vocal interfaces were
left to a later phase.  For the voice variant we therefore evaluate only its visual-manual parts (the
glances it still needs) - an illustration of the HMI trade-off, not an NHTSA assessment of a voice system.

The task model (DA-20) is a designed illustration, not measured data: per-step base glance durations, an age
factor, an individual factor and per-glance lognormal scatter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import labels as L
from . import regulations as R
from .config import PROPOSED, DmsConfig
from .scenarios import DATA_ORIGIN, Seg, timeline
from .system import onsets, run

AGE_FACTOR = {"18-24": 1.00, "25-39": 1.00, "40-54": 1.10, "55+": 1.25}   # DA-20


@dataclass(frozen=True)
class Participant:
    pid: str
    age_group: str
    sex: str
    factor: float              # individual glance-duration factor
    chars_per_glance: int      # characters typed per keyboard glance (touch variant)


def participants(seed: int = 2013) -> list[Participant]:
    rng = np.random.default_rng(seed)
    out = []
    for g in R.NHTSA_AGE_GROUPS:                      # VI.A.4 / VI.A.5: 6 per group, 3 men + 3 women
        for i in range(6):
            out.append(Participant(f"P{len(out) + 1:02d}", g, "M" if i < 3 else "F",
                                   float(AGE_FACTOR[g] * math.exp(0.12 * rng.standard_normal())),
                                   int(rng.integers(2, 5))))
    return out


# Task: change the navigation destination to a new street address while driving (DA-20)
ADDRESS_CHARS = 14                    # characters typed before the wanted suggestion appears


def task_glances(variant: str, p: Participant, rng: np.random.Generator) -> list[tuple[str, float]]:
    """Sequence of (label, duration) for one task performance: device glances interleaved with road glances."""
    dev: list[tuple[str, float]] = []        # (label, base duration)

    def g(label: str, base: float):
        dev.append((label, base))
    if variant == "touch_address":
        g(L.INFOTAINMENT, 1.2)                               # open navigation
        g(L.INFOTAINMENT, 1.0)                               # tap the search field
        for _ in range(math.ceil(ADDRESS_CHARS / p.chars_per_glance)):
            g(L.INFOTAINMENT, 1.6)                           # on-screen keyboard
        g(L.INFOTAINMENT, 1.8)                               # read the suggestion list, select
        g(L.INFOTAINMENT, 1.1)                               # confirm "Start"
    elif variant == "touch_favourite":
        g(L.INFOTAINMENT, 1.2)                               # open navigation
        g(L.INFOTAINMENT, 1.0)                               # tap favourites
        g(L.INFOTAINMENT, 1.4)                               # pick the stored entry
        g(L.INFOTAINMENT, 1.0)                               # start
    elif variant == "voice_address":
        if rng.random() < 0.4:
            g(L.INSTRUMENT, 0.6)                             # look for the push-to-talk button on the wheel
        for attempt in range(2 if rng.random() < 0.12 else 1):   # 12 % recognition errors -> one retry
            if rng.random() < 0.6:
                g(L.INFOTAINMENT, 1.0)                       # read the recognised text
            g(L.INFOTAINMENT, 1.2)                           # check the proposed destination
        if rng.random() < 0.3:
            g(L.INFOTAINMENT, 0.8)                           # tap "Start" instead of saying "yes"
    else:
        raise ValueError(variant)
    seq: list[tuple[str, float]] = []
    for i, (lab, base) in enumerate(dev):
        if i:
            seq.append((L.ROAD, float(rng.uniform(0.8, 1.6))))
        seq.append((lab, float(base * p.factor * math.exp(0.20 * rng.standard_normal()))))
    return seq


def glance_metrics(seq: list[tuple[str, float]]) -> dict:
    d = np.array([dur for lab, dur in seq if lab != L.ROAD])
    n = len(d)
    n_long = int((d > R.NHTSA_GLANCE_S).sum())
    allowed = math.ceil(R.NHTSA_LONG_GLANCE_SHARE * n - 1e-9)       # "15 percent (rounded up)"
    return {"n_glances": n, "mean_glance_s": round(float(d.mean()), 3), "n_glances_gt2s": n_long,
            "allowed_gt2s": allowed, "teort_s": round(float(d.sum()), 3),
            "crit_a": n_long <= allowed, "crit_b": float(d.mean()) <= R.NHTSA_GLANCE_S + 1e-9,
            "crit_c": float(d.sum()) <= R.NHTSA_TEORT_S + 1e-9}


def acceptance(rows: pd.DataFrame) -> dict:
    res = {}
    for c in ("crit_a", "crit_b", "crit_c"):
        k = int(rows[c].sum())
        res[f"n_meet_{c[-1]}"] = k
        res[f"pass_{c[-1]}"] = k >= R.NHTSA_PASS_MIN
    res["task_acceptable"] = res["pass_a"] and res["pass_b"] and res["pass_c"]
    return res


def dms_alerts(seq: list[tuple[str, float]], cfg: DmsConfig = PROPOSED, speed: float = 80.0) -> int:
    """Embed the task in a drive (60 s attentive before, 15 s after) and count DMS distraction alerts."""
    segs = [Seg(60.0, speed=speed)] + [Seg(dur, lab, blinks=lab == L.ROAD) for lab, dur in seq] + [Seg(15.0)]
    out = run(timeline(segs, speed), cfg, 0.1)
    return int((onsets(out, "acoustic", "distraction") > 60.0).sum())


def evaluate(seed: int = 2013, variants=("touch_address", "voice_address", "touch_favourite")) -> tuple[pd.DataFrame, pd.DataFrame]:
    ps = participants(seed)
    rows = []
    for vi, v in enumerate(variants):
        rng = np.random.default_rng(seed + 101 * (vi + 1))
        for p in ps:
            seq = task_glances(v, p, rng)
            rows.append({"variant": v, "pid": p.pid, "age_group": p.age_group, "sex": p.sex,
                         **glance_metrics(seq), "dms_alerts": dms_alerts(seq), "data_origin": DATA_ORIGIN})
    per = pd.DataFrame(rows)
    summ = []
    for v, g in per.groupby("variant", sort=False):
        a = acceptance(g)
        summ.append({"variant": v, **a, "median_teort_s": round(float(g.teort_s.median()), 2),
                     "max_teort_s": round(float(g.teort_s.max()), 2),
                     "median_mean_glance_s": round(float(g.mean_glance_s.median()), 2),
                     "share_with_dms_alert": round(float((g.dms_alerts > 0).mean()), 3), "data_origin": DATA_ORIGIN})
    return per, pd.DataFrame(summ)
