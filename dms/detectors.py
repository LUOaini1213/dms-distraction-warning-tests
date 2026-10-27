"""Per-frame detectors.  All timers are integer frame counters; seconds = frames * cfg.dt.

LongGlance   ADDW Annex I 3.3.2.1-3.3.2.4 / Euro NCAP DE 1.2.1: duration of the current counted glance.
Vats         Euro NCAP DE 1.2.2-1.2.3: counted glance time inside a sliding 30 s window, ended by a continuous
             forward-road gaze of T_VATS_reset (design assumption DA-06).
EyesOffRoad  Euro NCAP DE 1.4.3.1: continuous gaze away from the forward road (any region) for the ADAS coupling.
Drowsiness   PERCLOS (DDAW Part 2 5.2.1(a) names it; thresholds are design assumptions) and eye-closure runs
             for microsleep / sleep / unresponsive (Euro NCAP DE 1.3.3-1.3.5).
Availability ADDW 3.5 failure / temporary limitation, Euro NCAP DE 1.1.1.2 "inform if not functional".
"""
from __future__ import annotations

from collections import deque

from . import labels as L
from .config import DmsConfig

NOMINAL, DEGRADED, UNAVAILABLE, FAILED = "nominal", "degraded", "unavailable", "failed"


class LongGlance:
    """Frames in the current counted glance.

    * an interruption of <= tol frames (any label) does not reset the glance and is counted as part of it
      (ADDW 3.3.2.4: no reset for image artefacts or a short 'in, out and back in', tolerance >= 50 ms);
    * more than tol frames of a known non-counted label (road, mirror, ...) ends the glance;
    * 'unknown' longer than tol freezes the count: the time is not measurable (ADDW 3.3.2.3 counts the time
      'as long as the ADDW system is activated and able to measure it'), but the glance is not discarded.
    """
    __slots__ = ("counted", "unknown_counts", "tol", "n", "gap", "gap_known", "active")

    def __init__(self, counted: frozenset, unknown_counts: bool, tol_frames: int):
        self.counted, self.unknown_counts, self.tol = counted, unknown_counts, tol_frames
        self.reset()

    def reset(self) -> None:
        self.n = self.gap = self.gap_known = 0
        self.active = False

    def update(self, label: str) -> int:
        if label in self.counted or (self.unknown_counts and label == L.UNKNOWN):
            if self.active:
                if 0 < self.gap <= self.tol:
                    self.n += self.gap
            else:
                self.active, self.n = True, 0
            self.n += 1
            self.gap = self.gap_known = 0
        elif self.active:
            self.gap += 1
            if label != L.UNKNOWN:
                self.gap_known += 1
                if self.gap_known > self.tol:
                    self.reset()
        return self.n


class Vats:
    """Counted frames within the last `window` frames; the episode is cleared after `reset_n` frames of
    continuous road gaze.  'unknown' neither adds nor breaks the road run; mirror / instrument break it."""
    __slots__ = ("counted", "unknown_counts", "win", "reset_n", "buf", "pos", "sum", "road_run")

    def __init__(self, counted: frozenset, unknown_counts: bool, window_frames: int, reset_frames: int):
        self.counted, self.unknown_counts = counted, unknown_counts
        self.win, self.reset_n = window_frames, reset_frames
        self.reset()

    def reset(self) -> None:
        self.buf = bytearray(self.win)
        self.pos = self.sum = self.road_run = 0

    def update(self, label: str) -> int:
        c = 1 if (label in self.counted or (self.unknown_counts and label == L.UNKNOWN)) else 0
        self.sum += c - self.buf[self.pos]
        self.buf[self.pos] = c
        self.pos = self.pos + 1 if self.pos + 1 < self.win else 0
        if label == L.ROAD:
            self.road_run += 1
            if self.road_run >= self.reset_n and self.sum:
                self.buf = bytearray(self.win)          # attention back on the road long enough: episode ends
                self.sum = 0
        elif label != L.UNKNOWN:
            self.road_run = 0
        return self.sum


class EyesOffRoad:
    """Continuous eyes-off-road frames (any non-road label, short road flickers <= tol tolerated) and the
    continuous road-gaze run (blinks do not break it)."""
    __slots__ = ("tol", "eor", "road")

    def __init__(self, tol_frames: int):
        self.tol = tol_frames
        self.reset()

    def reset(self) -> None:
        self.eor = self.road = 0

    def update(self, label: str) -> tuple[int, int]:
        if label == L.ROAD:
            self.road += 1
            if self.road > self.tol:
                self.eor = 0
        elif label != L.UNKNOWN:
            self.eor += 1
            self.road = 0
        return self.eor, self.road


class Drowsiness:
    """PERCLOS over a sliding window of valid frames, eye-closure run length and the drowsiness level.

    level 0 alert, 1 information (PERCLOS >= l1), 2 warning (PERCLOS >= l2, or >= `microsleep_to_l2`
    microsleeps within `microsleep_memory_s`).  Hysteresis on exit.  PERCLOS is only reported once the window
    is full and at least half of it is valid (learning phase, Euro NCAP DE 1.1 allows up to 10 min).
    """

    def __init__(self, cfg: DmsConfig):
        self.thr = cfg.eyelid_closed
        self.win = cfg.n(cfg.perclos_window_s)
        self.l1, self.l1x, self.l2, self.l2x = cfg.perclos_l1, cfg.perclos_l1_exit, cfg.perclos_l2, cfg.perclos_l2_exit
        self.n_ms = cfg.n(cfg.t_microsleep_s)
        self.n_mem = cfg.n(cfg.microsleep_memory_s)
        self.ms_to_l2 = cfg.microsleep_to_l2
        self.reset()

    def reset(self) -> None:
        self.buf = bytearray(self.win)       # 0 invalid, 1 open, 2 closed
        self.pos = self.closed = self.valid = self.count = self.closure = 0
        self.base = 0
        self.ms_events: deque[int] = deque()

    def update(self, k: int, eyelid: float, eyes_ok: bool) -> tuple[float, int, int]:
        v = 0
        if eyes_ok and eyelid == eyelid:      # not NaN
            v = 2 if eyelid >= self.thr else 1
        old = self.buf[self.pos]
        self.buf[self.pos] = v
        self.pos = self.pos + 1 if self.pos + 1 < self.win else 0
        self.closed += (v == 2) - (old == 2)
        self.valid += (v > 0) - (old > 0)
        if self.count < self.win:
            self.count += 1
        self.closure = self.closure + 1 if v == 2 else 0
        if self.closure == self.n_ms:
            self.ms_events.append(k)
        while self.ms_events and k - self.ms_events[0] > self.n_mem:
            self.ms_events.popleft()
        perclos = -1.0
        if self.count == self.win and self.valid * 2 >= self.win:
            perclos = self.closed / self.valid
            if self.base == 2:
                self.base = 2 if perclos >= self.l2x else (1 if perclos >= self.l1x else 0)
            elif self.base == 1:
                self.base = 2 if perclos >= self.l2 else (1 if perclos >= self.l1x else 0)
            else:
                self.base = 2 if perclos >= self.l2 else (1 if perclos >= self.l1 else 0)
        else:
            self.base = 0                    # not measurable: no drowsiness claim
        level = 2 if len(self.ms_events) >= self.ms_to_l2 else self.base
        return perclos, self.closure, level


class Availability:
    """Sensor availability (inputs per frame: camera_ok, face_detected, eyes_visible).

    FAILED       camera fault or no light for >= t_fail_confirm (ADDW 3.5.1.1 / 3.5.1.3): constant failure
                 tell-tale, held while the fault is present (3.5.1.4).
    UNAVAILABLE  face not detected (e.g. hand over a cluster-mounted camera): gaze and eyelid not usable.
    DEGRADED     face tracked but eyes not visible (sunglasses < 15 % transmittance): head pose only, so the
                 gaze label still exists for head ('owl') movements; eyelid not usable (Euro NCAP DE
                 definition of a degraded system).
    The driver is informed after t_inform of non-nominal status (Euro NCAP DE 1.1.1.2: within 10 s; visual
    for as long as it lasts, audible at most once per journey; ADDW 3.5.2.2 information on the limitation).
    """

    def __init__(self, cfg: DmsConfig):
        self.n_fail, self.n_inform, self.n_recover = cfg.n(cfg.t_fail_confirm_s), cfg.n(cfg.t_inform_s), cfg.n(cfg.t_recover_s)
        self.reset()

    def reset(self) -> None:
        self.fault_run = self.ok_run = self.limited_run = self.valid_run = 0
        self.failed = self.info = self.audible_used = False

    def update(self, camera_ok: bool, face: bool, eyes: bool) -> tuple[str, bool, bool]:
        """Returns (status, info_visual, info_audible_pulse)."""
        if camera_ok:
            self.fault_run, self.ok_run = 0, self.ok_run + 1
            if self.failed and self.ok_run >= self.n_recover:
                self.failed = False
        else:
            self.fault_run, self.ok_run = self.fault_run + 1, 0
            if self.fault_run >= self.n_fail:
                self.failed = True
        if self.failed:
            status = FAILED
        elif not camera_ok or not face:
            status = UNAVAILABLE
        elif not eyes:
            status = DEGRADED
        else:
            status = NOMINAL
        if status == NOMINAL:
            self.limited_run, self.valid_run = 0, self.valid_run + 1
            if self.info and self.valid_run >= self.n_recover:
                self.info = False
        else:
            self.limited_run, self.valid_run = self.limited_run + 1, 0
        pulse = False
        if status in (UNAVAILABLE, DEGRADED) and self.limited_run >= self.n_inform and not self.info:
            self.info = True
            if not self.audible_used:
                self.audible_used, pulse = True, True
        return status, self.info and status != FAILED, pulse
