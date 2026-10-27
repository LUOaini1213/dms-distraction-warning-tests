"""Warning state machine: escalation, termination, suppression, arbitration and the ADAS coupling.

Distraction alert (one per episode)
  IDLE --pre-alert--> VISUAL --trigger--> ACOUSTIC (+visual) --trigger persists t_haptic--> HAPTIC (+acoustic+visual)
  * trigger = long glance >= t_long (ADDW 3.3.2.1/3.3.2.2, Euro NCAP DE 1.2.1) or VATS latched (DE 1.2.2)
  * the "warning start" is the acoustic onset (ADDW 3.4.1.1: "considered to start when the audio or haptic
    warning is presented"); Euro NCAP DE 1.4.1 asks for visual + audible/haptic immediately at classification
  * a new alert within repeat_window of the last one starts directly at HAPTIC (ADDW 3.4.1.2 adaptation)
  * the alert ends when the trigger ceases (ADDW 3.4.1.1), but not before min_acoustic (ADDW 3.4.3.4)
Suppression
  * speed < v_distraction, calibration running, driver switched ADDW warnings off for this journey
    (ADDW 3.1.2 / 3.1.6), sensor unavailable: no distraction alert; timers keep running (ADDW 3.3.2.3)
  * another ADAS warns of imminent danger: DMS acoustic / haptic removed for distraction and drowsiness
    alerts (ADDW 3.1.5), restored when it ends
Arbitration (Regulation (EU) 2019/2144 Art. 6(3): no separate, concurrent or confusing prompts)
  unresponsive > sleep > microsleep > distraction alert > drowsiness alert > distraction visual pre-alert
  > drowsiness info > DMS-limited info > visual hold.  Exactly one prompt source per frame; the failure
  tell-tale is a separate constant indicator.
"""
from __future__ import annotations

from . import labels as L
from .config import DmsConfig
from .detectors import DEGRADED, NOMINAL

URGENCY = {"": 0, "distraction_hold": 1, "dms_limited": 1, "drowsy_info": 1, "pre_alert": 1,
           "drowsy": 2, "distraction_long_glance": 2, "distraction_vats": 2,
           "microsleep": 3, "sleep": 4, "unresponsive": 5}
NEG = -10 ** 9


class WarningManager:
    def __init__(self, cfg: DmsConfig):
        self.cfg = cfg
        n = cfg.n
        self.N_long, self.N_long_low = n(cfg.t_long_s), n(cfg.t_long_low_s) if cfg.t_long_low_s else n(cfg.t_long_s)
        self.N_pre, self.N_vats, self.N_vats_pre = n(cfg.t_pre_s), n(cfg.vats_threshold_s), n(cfg.vats_pre_s)
        self.N_vats_reset = n(cfg.vats_reset_s)
        self.N_haptic, self.N_repeat = n(cfg.t_haptic_after_s), n(cfg.repeat_window_s)
        self.N_min_ac, self.N_vis_hold = n(cfg.min_acoustic_s), n(cfg.visual_hold_s)
        self.N_unresp_gaze, self.N_resp_road = n(cfg.t_unresp_gaze_s), n(1.0)
        self.N_ms, self.N_sleep, self.N_unresp_eyes = n(cfg.t_microsleep_s), n(cfg.t_sleep_s), n(cfg.t_unresp_eyes_s)
        self.N_ms_hold, self.N_l2_alert, self.N_l2_repeat = n(cfg.microsleep_hold_s), n(cfg.l2_alert_s), n(cfg.l2_repeat_s)
        self.N_fss_on, self.N_fss_off_road, self.N_fss_off_state = n(cfg.t_fss_on_s), n(cfg.t_fss_off_road_s), n(cfg.t_fss_off_state_s)
        self.reset()

    def reset(self) -> None:
        self.state = 0                 # 0 idle, 1 visual pre-alert, 2 acoustic, 3 haptic
        self.src = ""
        self.ac_start = NEG
        self.last_end = NEG
        self.vats_latched = False
        self.returned = True
        self.unresp_gaze = False
        self.vis_hold_until = NEG
        self.ms_hold_until = NEG
        self.l2_next = 0
        self.l2_until = NEG
        self.fss = False
        self.fss_state = False
        self.fss_state_end = None

    def update(self, k: int, speed: float, label: str, ld_n: int, vats_n: int, eor_n: int, road_run: int,
               closure: int, level: int, status: str, info: bool, info_pulse: bool, calibrated: bool,
               addw_off: bool, adas_warn: bool) -> tuple:
        cfg = self.cfg
        gaze_ok = status == NOMINAL or status == DEGRADED
        eyes_ok = status == NOMINAL
        # ------------------------------------------------------------ distraction episode
        gate = speed >= cfg.v_distraction_kph and calibrated and not addw_off and gaze_ok
        n_long = self.N_long_low if (cfg.t_long_low_s and speed < cfg.v_high_kph) else self.N_long
        if cfg.vats_enabled and vats_n >= self.N_vats:
            self.vats_latched = True
        if self.vats_latched and road_run >= self.N_vats_reset:
            self.vats_latched = False
        trig_ld = ld_n >= n_long
        trig = gate and (trig_ld or self.vats_latched)
        pre = gate and (ld_n >= self.N_pre or (cfg.vats_enabled and vats_n >= self.N_vats_pre))
        if trig:
            if self.state < 2:
                self.state = 3 if k - self.last_end <= self.N_repeat else 2
                self.ac_start = k
                self.src = "distraction_long_glance" if trig_ld else "distraction_vats"
                self.returned = False
            elif self.state == 2 and k - self.ac_start >= self.N_haptic:
                self.state = 3
        elif self.state >= 2:
            if not gate or k - self.ac_start >= self.N_min_ac:
                self.state, self.last_end, self.vis_hold_until = 0, k, k + self.N_vis_hold
        elif pre:
            self.state, self.src = 1, "pre_alert"
        else:
            self.state = 0
        # responsiveness after a distraction alert (Euro NCAP DE 1.3.5)
        if label == L.ROAD:
            self.returned = True
        if self.state >= 2 and not self.returned and k - self.ac_start >= self.N_unresp_gaze:
            self.unresp_gaze = True
        if self.unresp_gaze and road_run >= self.N_resp_road:
            self.unresp_gaze = False
        # ------------------------------------------------------------ drowsiness / eye closure
        eye_gate = eyes_ok and speed >= cfg.v_distraction_kph       # microsleep/sleep at >= 20 km/h (DE 1.1)
        if eye_gate and closure >= self.N_ms:
            self.ms_hold_until = k + self.N_ms_hold
        ms = k <= self.ms_hold_until
        sleep = eye_gate and closure >= self.N_sleep
        unresp = self.unresp_gaze or (eye_gate and closure >= self.N_unresp_eyes)
        dgate = eyes_ok and speed >= cfg.v_drowsy_kph
        blocked = self.state >= 2 or ms or sleep or unresp   # a due drowsiness alert waits (deferred, not dropped)
        if dgate and level >= 2:
            if k >= self.l2_next and not blocked:
                self.l2_until, self.l2_next = k + self.N_l2_alert, k + self.N_l2_repeat
        elif level < 2:
            self.l2_next = 0
        drowsy_alert = dgate and level >= 2 and k < self.l2_until
        drowsy_info = dgate and level >= 1
        # ------------------------------------------------------------ arbitration (one prompt at a time)
        v = a = h = False
        if unresp:
            src, v, a, h = "unresponsive", True, True, True
        elif sleep:
            src, v, a, h = "sleep", True, True, True
        elif ms:
            src, v, a, h = "microsleep", True, True, True
        elif self.state >= 2:
            src, v, a, h = self.src, True, True, self.state == 3
        elif drowsy_alert:
            src, v, a = "drowsy", True, True
        elif self.state == 1:
            src, v = "pre_alert", True
        elif drowsy_info:
            src, v = "drowsy_info", True
        elif info:
            src, v, a = "dms_limited", True, info_pulse
        elif k <= self.vis_hold_until:
            src, v = "distraction_hold", True
        else:
            src = ""
        if adas_warn and (src.startswith("distraction") or src == "drowsy" or src == "pre_alert"):
            a = h = False                                   # ADDW 3.1.5
        # ------------------------------------------------------------ ADAS (FCW/AEB) sensitivity coupling
        state_cause = self.state >= 2 or ms or sleep or unresp or level >= 2
        if state_cause:
            self.fss, self.fss_state, self.fss_state_end = True, True, None
        elif eor_n >= self.N_fss_on:
            self.fss = True
        elif self.fss:
            if self.fss_state:
                if self.fss_state_end is None:
                    self.fss_state_end = k
                if k - self.fss_state_end >= self.N_fss_off_state and road_run >= self.N_fss_off_road:
                    self.fss = self.fss_state = False
            elif road_run >= self.N_fss_off_road:
                self.fss = False
        return v, a, h, src, URGENCY.get(src, 0), self.fss, unresp
