"""SYNTHETIC scenario generator.  Every DataFrame produced here carries attrs["data_origin"] = "SYNTHETIC" and
every results file written from it has a data_origin column with that value.  Nothing here is measured data.

timeline()   deterministic segment-by-segment scenarios (regulation and boundary test cases)
natural()    natural-looking driving: semi-Markov glance model + blinks + perception noise + speed profile,
             with injected distraction episodes whose ground truth is known (false-alarm study).
All behavioural parameters are design assumption DA-19 (glance model, perception noise, driving-related context
events).  natural() returns observed labels (`gaze`), ground-truth location (`gt_gaze`) and ground-truth
activity (`gt_activity`: road / driving / non_driving) - a shoulder check is off_road by location but driving
by activity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import labels as L
from .config import PROPOSED

DATA_ORIGIN = "SYNTHETIC"
EYELID_OPEN = 0.10          # closure fraction of open eyes
EYELID_DOWN = 0.45          # eyes look more closed when gazing down (phone, lap, console) - perception artefact
DOWNWARD = frozenset({L.PHONE, L.OFF_ROAD, L.INFOTAINMENT, L.INSTRUMENT})
COLS = ("t", "speed_kph", "gaze", "eyelid", "camera_ok", "face_detected", "eyes_visible", "adas_warning", "addw_off")


@dataclass
class Seg:
    dur: float
    gaze: str = L.ROAD
    eyelid: float | None = None        # None -> default for the gaze direction
    speed: float | None = None         # None -> keep the current speed
    speed_end: float | None = None     # linear ramp inside the segment
    camera_ok: bool = True
    face: bool = True
    eyes: bool = True
    adas: bool = False
    addw_off: bool = False
    blinks: bool = True                # periodic blinks while looking at the road


def frames(dur: float, dt: float = PROPOSED.dt) -> int:
    return int(round(dur / dt))


def timeline(segs: list[Seg], speed: float = 50.0, dt: float = PROPOSED.dt, blink_period_s: float = 3.5,
             blink_frames: int = 4, phase_s: float = 0.0) -> pd.DataFrame:
    n_tot = sum(frames(s.dur, dt) for s in segs)
    gaze = np.empty(n_tot, dtype=object)
    eyelid = np.empty(n_tot)
    spd = np.empty(n_tot)
    flags = {c: np.empty(n_tot, dtype=bool) for c in ("camera_ok", "face_detected", "eyes_visible", "adas_warning", "addw_off")}
    blink_ok = np.zeros(n_tot, dtype=bool)
    i, v = 0, speed
    for s in segs:
        n = frames(s.dur, dt)
        if s.speed is not None:
            v = s.speed
        v1 = v if s.speed_end is None else s.speed_end
        spd[i:i + n] = np.linspace(v, v1, n, endpoint=False) if n else []
        v = v1
        gaze[i:i + n] = s.gaze
        eyelid[i:i + n] = s.eyelid if s.eyelid is not None else (EYELID_DOWN if s.gaze in DOWNWARD else EYELID_OPEN)
        flags["camera_ok"][i:i + n] = s.camera_ok
        flags["face_detected"][i:i + n] = s.face
        flags["eyes_visible"][i:i + n] = s.eyes
        flags["adas_warning"][i:i + n] = s.adas
        flags["addw_off"][i:i + n] = s.addw_off
        blink_ok[i:i + n] = s.blinks and s.gaze == L.ROAD
        i += n
    t = np.arange(n_tot) * dt
    per = int(round(blink_period_s / dt))
    ph = int(round(phase_s / dt))
    blink = blink_ok & (((np.arange(n_tot) + ph) % per) < blink_frames)
    gaze[blink] = L.UNKNOWN
    eyelid[blink] = 1.0
    df = pd.DataFrame({"t": t, "speed_kph": spd, "gaze": gaze, "eyelid": eyelid, **flags})
    df.attrs["data_origin"] = DATA_ORIGIN
    return df


# =============================================================================== natural driving
@dataclass
class NaturalParams:
    """DA-19: synthetic glance behaviour, perception noise and context events (not calibrated to naturalistic data)."""
    road_median_s: float = 3.5
    road_sigma: float = 0.6
    road_min_s: float = 0.6
    mix: dict = field(default_factory=lambda: {L.MIRROR: 0.32, L.INSTRUMENT: 0.33, L.INFOTAINMENT: 0.17, L.OFF_ROAD: 0.18})
    median_s: dict = field(default_factory=lambda: {L.MIRROR: 0.65, L.INSTRUMENT: 0.55, L.INFOTAINMENT: 0.95, L.OFF_ROAD: 0.85})
    sigma: dict = field(default_factory=lambda: {L.MIRROR: 0.35, L.INSTRUMENT: 0.35, L.INFOTAINMENT: 0.40, L.OFF_ROAD: 0.45})
    glance_max_s: float = 6.0
    blink_per_min: float = 17.0
    blink_median_s: float = 0.13
    blink_sigma: float = 0.30
    flicker_per_min: float = 4.0       # 1-3 frame label flicker to a random other label
    dropout_per_min: float = 0.6       # tracking dropouts -> 'unknown'
    dropout_median_s: float = 0.35
    misclass: dict = field(default_factory=lambda: {L.MIRROR: (L.OFF_ROAD, 0.05), L.INSTRUMENT: (L.INFOTAINMENT, 0.03)})
    eyelid_noise: float = 0.04
    # DA-19 driving-related context events (rates per second, only inside their speed band)
    lane_change_rate: float = 1 / 90.0        # >= 60 km/h: mirror, mirror, shoulder check
    merge_rate: float = 1 / 600.0             # 40-80 km/h: long look over the shoulder for a gap
    junction_rate: float = 1 / 150.0          # 15-55 km/h: side-window scanning left / right
    nav_rate: float = 1 / 240.0               # route-guidance checks on the centre display
    acc_rate: float = 1 / 600.0               # >= 90 km/h: setting ACC on the steering wheel / cluster


# ground-truth activity of a frame: what the driver is doing, independent of where the label says he looks
ACT_ROAD, ACT_DRIVING, ACT_NON_DRIVING = "road", "driving", "non_driving"


@dataclass
class Episode:
    eid: int
    kind: str          # "long_glance" | "phone_vats"
    t0: float
    t1: float
    label: str


def _speed_profile(rng: np.random.Generator, n: int, dt: float) -> np.ndarray:
    """Blocks of urban (with a stop), rural and highway driving; linear 6 s transitions."""
    knots_t, knots_v, t = [0.0], [0.0], 0.0
    while t < n * dt:
        kind = rng.choice(["urban", "rural", "highway"], p=[0.4, 0.35, 0.25])
        dur = rng.uniform(120, 240)
        if kind == "urban":
            v = rng.uniform(35, 50)
            stop_at = t + rng.uniform(0.4, 0.7) * dur
            for tt, vv in ((t + 6, v), (stop_at - 6, v), (stop_at, 0.0), (stop_at + rng.uniform(15, 30), 0.0)):
                knots_t.append(tt); knots_v.append(vv)
            knots_t.append(knots_t[-1] + 6); knots_v.append(v)
        else:
            v = rng.uniform(60, 90) if kind == "rural" else rng.uniform(100, 120)
            knots_t.append(t + 6); knots_v.append(v)
        t += dur
        knots_t.append(t); knots_v.append(knots_v[-1])
    return np.interp(np.arange(n) * dt, knots_t, knots_v)


def _lognormal(rng, median, sigma, size=None):
    return median * np.exp(sigma * rng.standard_normal(size))


def natural(duration_s: float, seed: int, p: NaturalParams | None = None, dt: float = PROPOSED.dt,
            inject: bool = True) -> tuple[pd.DataFrame, list[Episode]]:
    """One synthetic drive.  Returns the scenario (observed labels in `gaze`, ground truth in `gt_gaze`,
    `episode` = id of the injected distraction episode or -1) and the list of injected episodes."""
    p = p or NaturalParams()
    rng = np.random.default_rng(seed)
    n = frames(duration_s, dt)
    speed = _speed_profile(rng, n, dt)
    gt = np.empty(n, dtype=object)
    i = 0
    regions, probs = list(p.mix), np.array(list(p.mix.values()))
    act = np.empty(n, dtype=object)
    while i < n:                                   # semi-Markov: road dwell, then one glance
        d_road = max(p.road_min_s, _lognormal(rng, p.road_median_s, p.road_sigma))
        k = frames(d_road, dt)
        gt[i:i + k] = L.ROAD
        act[i:i + k] = ACT_ROAD
        i += k
        if i >= n:
            break
        r = regions[rng.choice(len(regions), p=probs)]
        d = min(p.glance_max_s, _lognormal(rng, p.median_s[r], p.sigma[r]))
        k = max(1, frames(d, dt))
        gt[i:i + k] = r
        act[i:i + k] = ACT_DRIVING if r in L.DRIVING_TASK else ACT_NON_DRIVING
        i += k
    # ---------------------------------------------------------------- driving-related context events (DA-19)
    def put(t0: float, seq: list[tuple[str, float, str]]):
        j = frames(t0, dt)
        for lab, dur, a in seq:
            k = frames(dur, dt)
            gt[j:j + k] = lab
            act[j:j + k] = a
            j += k
        if j < n:                                  # always end on the road
            gt[j:j + frames(1.0, dt)] = L.ROAD
            act[j:j + frames(1.0, dt)] = ACT_ROAD
    D, ND, RD = ACT_DRIVING, ACT_NON_DRIVING, ACT_ROAD
    events = (
        (p.lane_change_rate, (60, 200), lambda: [(L.MIRROR, 0.7, D), (L.ROAD, 0.6, RD), (L.MIRROR, 0.9, D), (L.ROAD, 0.4, RD),
                                                 (L.OFF_ROAD, rng.uniform(0.5, 1.0), D)]),
        (p.merge_rate, (40, 80), lambda: [(L.MIRROR, 1.2, D), (L.ROAD, 0.4, RD), (L.OFF_ROAD, rng.uniform(1.5, 3.0), D),
                                          (L.ROAD, 0.5, RD), (L.MIRROR, 0.8, D)]),
        (p.junction_rate, (15, 55), lambda: [(L.OFF_ROAD, rng.uniform(0.8, 1.6), D), (L.ROAD, 0.5, RD),
                                             (L.OFF_ROAD, rng.uniform(0.8, 1.6), D), (L.ROAD, 0.5, RD),
                                             (L.OFF_ROAD, rng.uniform(0.6, 1.2), D)]),
        (p.nav_rate, (15, 200), lambda: [x for _ in range(3) for x in ((L.INFOTAINMENT, rng.uniform(0.8, 1.5), ND),
                                                                       (L.ROAD, rng.uniform(1.0, 2.5), RD))]),
        (p.acc_rate, (90, 200), lambda: [(L.INSTRUMENT, rng.uniform(1.5, 2.5), D)]),
    )
    for rate, (v_lo, v_hi), make in events:
        t = rng.exponential(1 / rate)
        while t < duration_s - 15.0:
            v = speed[frames(t, dt)]
            if v_lo <= v <= v_hi:
                put(t, make())
            t += rng.exponential(1 / rate)
    episode = np.full(n, -1, dtype=int)
    eps: list[Episode] = []
    if inject:
        for eid, kind in enumerate(("long_glance", "phone_vats")):
            dur = rng.uniform(4.0, 6.0) if kind == "long_glance" else rng.uniform(20.0, 35.0)
            placed = False
            for _ in range(500):                   # place it where the car moves at >= 40 km/h throughout
                t0 = rng.uniform(120.0, duration_s - dur - 20.0)
                a, b = frames(t0, dt), frames(t0 + dur, dt)
                if speed[a - frames(5, dt):b + frames(5, dt)].min() >= 40.0 and \
                        (episode[max(0, a - frames(30, dt)):b + frames(30, dt)] < 0).all():
                    placed = True
                    break
            if not placed:
                continue
            gt[a - frames(4.0, dt):a] = L.ROAD     # attentive just before (Euro NCAP-like preceding road gaze)
            act[a - frames(4.0, dt):a] = ACT_ROAD
            if kind == "long_glance":
                lab = L.INFOTAINMENT if rng.random() < 0.5 else L.PHONE
                gt[a:b] = lab
                act[a:b] = ACT_NON_DRIVING
            else:
                lab = L.PHONE
                j = a
                while j < b:
                    g = frames(min(3.0, _lognormal(rng, 1.8, 0.25)), dt)
                    gt[j:min(b, j + g)] = L.PHONE
                    act[j:min(b, j + g)] = ACT_NON_DRIVING
                    j += g
                    r = frames(rng.uniform(0.5, 1.4), dt)
                    gt[j:min(b, j + r)] = L.ROAD
                    act[j:min(b, j + r)] = ACT_ROAD
                    j += r
            gt[b:b + frames(3.0, dt)] = L.ROAD
            act[b:b + frames(3.0, dt)] = ACT_ROAD
            episode[a:b] = eid
            eps.append(Episode(eid, kind, a * dt, b * dt, lab))
    # ---------------------------------------------------------------- observation = truth + noise
    obs = gt.copy()
    # whole-glance misclassification
    change = np.flatnonzero(np.concatenate(([True], gt[1:] != gt[:-1])))
    bounds = np.append(change, n)
    for s, e in zip(bounds[:-1], bounds[1:]):
        lab = gt[s]
        if lab in p.misclass and rng.random() < p.misclass[lab][1]:
            obs[s:e] = p.misclass[lab][0]
    eyelid = np.where(np.isin(gt, list(DOWNWARD)), EYELID_DOWN, EYELID_OPEN) + p.eyelid_noise * rng.standard_normal(n)
    minutes = duration_s / 60.0
    for _ in range(rng.poisson(p.blink_per_min * minutes)):          # blinks: eyes closed, gaze unknown
        s = rng.integers(0, n)
        k = max(1, frames(_lognormal(rng, p.blink_median_s, p.blink_sigma), dt))
        obs[s:s + k] = L.UNKNOWN
        eyelid[s:s + k] = 1.0
    others = np.array(L.REGIONS[:-1], dtype=object)
    for _ in range(rng.poisson(p.flicker_per_min * minutes)):        # label flicker
        s = rng.integers(0, n)
        obs[s:s + rng.integers(1, 4)] = others[rng.integers(0, len(others))]
    for _ in range(rng.poisson(p.dropout_per_min * minutes)):        # tracking dropouts
        s = rng.integers(0, n)
        k = max(1, frames(_lognormal(rng, p.dropout_median_s, 0.5), dt))
        obs[s:s + k] = L.UNKNOWN
    eyelid = np.clip(eyelid, 0.0, 1.0)
    df = pd.DataFrame({"t": np.arange(n) * dt, "speed_kph": speed, "gaze": obs, "eyelid": eyelid,
                       "camera_ok": True, "face_detected": True, "eyes_visible": True,
                       "adas_warning": False, "addw_off": False, "gt_gaze": gt, "gt_activity": act, "episode": episode})
    df.attrs["data_origin"] = DATA_ORIGIN
    return df, eps
