"""DmsSystem: one call per frame, as an ECU task would run it.  run() drives it over a scenario DataFrame.

Scenario columns (dms/scenarios.py): t, speed_kph, gaze, eyelid, camera_ok, face_detected, eyes_visible,
adas_warning, addw_off.  Each run() is one journey: everything is reset first (ADDW 3.1.6, DDAW 3.1.3).

Privacy by design (Regulation (EU) 2019/2144 Art. 6(3), ADDW Annex I 2.3, DDAW Annex I 2.3.1): the system
keeps no identity and only fixed-size rolling buffers (VATS 30 s, PERCLOS 60 s, microsleep times 5 min);
run() returns outputs to the caller and stores nothing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import PROPOSED, DmsConfig
from .detectors import Availability, Drowsiness, EyesOffRoad, LongGlance, Vats
from .warning import WarningManager

OUT_COLS = ("visual", "acoustic", "haptic", "source", "urgency", "fss", "ef_request", "info", "failure",
            "status", "glance_s", "vats_s", "eor_s", "perclos", "closure_s", "drowsy_level")


class DmsSystem:
    def __init__(self, cfg: DmsConfig = PROPOSED):
        self.cfg = cfg
        tol = cfg.n(cfg.gap_tol_s) if cfg.gap_tol_s > 0 else 0   # 0.3 s -> 9 frames, 0.05 s -> 2 frames
        self.tol = tol
        self.ld = LongGlance(cfg.counted_regions, cfg.unknown_counts, tol)
        self.vats = Vats(cfg.counted_regions, cfg.unknown_counts, cfg.n(cfg.vats_window_s), cfg.n(cfg.vats_reset_s))
        self.eor = EyesOffRoad(tol)
        self.drowsy = Drowsiness(cfg)
        self.avail = Availability(cfg)
        self.warn = WarningManager(cfg)
        self.n_calib = cfg.n(cfg.calib_time_s)
        self.reset_journey()

    def reset_journey(self) -> None:
        """Activation of the vehicle master control switch: normal operation mode again."""
        for part in (self.ld, self.vats, self.eor, self.drowsy, self.avail, self.warn):
            part.reset()
        self.k = 0
        self.calib = 0

    def buffer_sizes(self) -> dict[str, int]:
        return {"vats": len(self.vats.buf), "perclos": len(self.drowsy.buf), "microsleep": len(self.drowsy.ms_events)}

    def step(self, speed: float, gaze: str, eyelid: float, camera_ok: bool = True, face: bool = True,
             eyes: bool = True, adas_warn: bool = False, addw_off: bool = False) -> tuple:
        cfg, k = self.cfg, self.k
        status, info, pulse = self.avail.update(camera_ok, face, eyes)
        gaze_ok = status in ("nominal", "degraded")
        label = gaze if gaze_ok else "unknown"
        if speed >= cfg.calib_speed_kph and self.calib < self.n_calib:
            self.calib += 1
        ld_n = self.ld.update(label)
        vats_n = self.vats.update(label)
        eor_n, road_run = self.eor.update(label)
        perclos, closure, level = self.drowsy.update(k, eyelid, status == "nominal")
        v, a, h, src, urg, fss, ef = self.warn.update(
            k, speed, label, ld_n, vats_n, eor_n, road_run, closure, level, status, info, pulse,
            self.calib >= self.n_calib, addw_off, adas_warn)
        self.k = k + 1
        dt = cfg.dt
        return (v, a, h, src, urg, fss, ef, info, status == "failed", status,
                ld_n * dt, vats_n * dt, eor_n * dt, perclos, closure * dt, level)


def _delay(arr: np.ndarray, frames: int) -> np.ndarray:
    if frames <= 0:
        return arr
    out = np.empty_like(arr)
    out[:frames] = arr[0]
    out[frames:] = arr[:-frames]
    return out


def run(scn: pd.DataFrame, cfg: DmsConfig = PROPOSED, latency_s: float = 0.0, system: DmsSystem | None = None) -> pd.DataFrame:
    """Run one journey.  latency_s delays the perception signals (gaze, eyelid, availability flags) to model
    camera + classifier latency (design assumption DA-02); speed and ADAS flags are not delayed."""
    sysm = system or DmsSystem(cfg)
    sysm.reset_journey()
    lag = int(round(latency_s / cfg.dt))
    gaze = _delay(scn["gaze"].to_numpy(dtype=object), lag)
    eyelid = _delay(scn["eyelid"].to_numpy(dtype=float), lag)
    cam = _delay(scn["camera_ok"].to_numpy(dtype=bool), lag)
    face = _delay(scn["face_detected"].to_numpy(dtype=bool), lag)
    eyes = _delay(scn["eyes_visible"].to_numpy(dtype=bool), lag)
    speed = scn["speed_kph"].to_numpy(dtype=float)
    adas = scn["adas_warning"].to_numpy(dtype=bool)
    off = scn["addw_off"].to_numpy(dtype=bool)
    step = sysm.step
    rows = [step(speed[i], gaze[i], eyelid[i], cam[i], face[i], eyes[i], adas[i], off[i]) for i in range(len(scn))]
    out = pd.DataFrame.from_records(rows, columns=OUT_COLS)
    out.insert(0, "t", scn["t"].to_numpy())
    # decisions become available at the end of the frame (t_k + dt), see report section 4.1
    out.insert(1, "t_dec", scn["t"].to_numpy() + cfg.dt)
    out.attrs["data_origin"] = scn.attrs.get("data_origin", "SYNTHETIC")
    return out


def onsets(out: pd.DataFrame, column: str, source_prefix: str | None = None) -> np.ndarray:
    """Decision times (t_dec) of rising edges of a boolean output column, optionally only for one source."""
    x = out[column].to_numpy(dtype=bool)
    if source_prefix is not None:
        x = x & out["source"].astype(str).str.startswith(source_prefix).to_numpy()
    rise = np.flatnonzero(x & ~np.concatenate(([False], x[:-1])))
    return out["t_dec"].to_numpy()[rise]


def first_onset(out: pd.DataFrame, column: str, after: float = -np.inf, source_prefix: str | None = None) -> float:
    t = onsets(out, column, source_prefix)
    t = t[t > after]
    return float(t[0]) if len(t) else float("nan")
