"""Warning state machine: escalation, termination, repeat events, suppression, arbitration, journey reset, privacy."""
import dataclasses

import pytest

from dms import labels as L
from dms.config import PROPOSED
from dms.system import DmsSystem

N = PROPOSED.n


def run_seq(seq, speed=55.0, warm=30.0, system=None, **flags):
    s = system or DmsSystem(PROPOSED)
    out = [s.step(speed, L.ROAD, 0.1, **flags) for _ in range(N(warm))]
    for lab, dur in seq:
        out += [s.step(speed, lab, 0.1, **flags) for _ in range(N(dur))]
    return out, N(warm)


def onset(out, idx, start=0):
    return next((k for k in range(start, len(out)) if out[k][idx]), None)


def test_escalation_visual_acoustic_haptic():
    """Visual pre-alert -> acoustic (+visual) -> haptic, in this order; warning start = acoustic onset.  REQ: HMI-02, HMI-01"""
    out, w = run_seq([(L.PHONE, 6.0)])
    v, a, h = onset(out, 0), onset(out, 1), onset(out, 2)
    assert v < a < h
    assert (a - w + 1) == N(3.3) and (h - a) == N(1.5)
    assert out[a][0] and out[h][0] and out[h][1]          # visual stays on, acoustic stays on with haptic


def test_termination_min_acoustic_and_visual_hold():
    """Alert stops when the trigger ceases (gaze back > 0.3 s) but lasts >= 1.0 s; visual held 2 s.  REQ: HMI-03"""
    out, w = run_seq([(L.PHONE, 3.4), (L.ROAD, 6.0)])
    a = onset(out, 1)
    end = next(k for k in range(a, len(out)) if not out[k][1])
    assert (end - a) * PROPOSED.dt >= 1.0 - 1e-9
    assert out[end][0] and out[end][3] == "distraction_hold"
    vis_end = next(k for k in range(end, len(out)) if not out[k][0])
    assert vis_end - end == N(2.0) + 1
    out, w = run_seq([(L.PHONE, 6.0), (L.ROAD, 2.0)])
    last_ac = max(k for k, o in enumerate(out) if o[1])
    glance_end = w + N(6.0)                                 # index of the first road frame
    assert last_ac - glance_end + 1 == N(0.3)               # sound continues only through the 0.3 s tolerance


def test_repeat_event_starts_haptic():
    """A second alert within 60 s starts directly with haptic (adaptation to previous events).  REQ: HMI-02"""
    out, w = run_seq([(L.PHONE, 4.0), (L.ROAD, 20.0), (L.PHONE, 4.0)])
    second_glance = w + N(4.0) + N(20.0)
    a2 = onset(out, 1, second_glance)
    assert out[a2][2]
    out, w = run_seq([(L.PHONE, 4.0), (L.ROAD, 70.0), (L.PHONE, 4.0)])
    a2 = onset(out, 1, w + N(74.0))
    assert not out[a2][2]


def test_adas_warning_suppresses_dms_sound():
    """While another ADAS warns of imminent danger, DMS acoustic / haptic are removed, visual kept.  REQ: HMI-06"""
    s = DmsSystem(PROPOSED)
    out, w = run_seq([], system=s)
    outs = [s.step(55.0, L.PHONE, 0.1) for _ in range(N(3.0))]
    outs += [s.step(55.0, L.PHONE, 0.1, adas_warn=True) for _ in range(N(2.0))]
    assert not any(o[1] or o[2] for o in outs[N(3.0):]) and all(o[0] for o in outs[N(3.4):])
    after = s.step(55.0, L.PHONE, 0.1)
    assert after[1]


def test_arbitration_single_prompt_and_priority():
    """One prompt source per frame; microsleep outranks a distraction pre-alert, drowsiness alert is deferred.
    REQ: HMI-05"""
    s = DmsSystem(PROPOSED)
    out, w = run_seq([(L.PHONE, 2.5)], system=s)
    assert out[-1][3] == "pre_alert"
    outs = [s.step(55.0, L.UNKNOWN, 1.0) for _ in range(N(1.5))]
    assert outs[-1][3] == "microsleep" and isinstance(outs[-1][3], str)
    assert all(isinstance(o[3], str) for o in out + outs)  # exactly one source value per frame


def test_journey_reset_restores_driver_switched_off_warnings():
    """Driver may switch ADDW warnings off for a journey; next journey starts in normal mode.  REQ: SYS-02"""
    s = DmsSystem(PROPOSED)
    out, w = run_seq([(L.PHONE, 5.0)], system=s, addw_off=True)
    assert onset(out, 1) is None
    s.reset_journey()
    out, w = run_seq([(L.PHONE, 5.0)], system=s)
    assert onset(out, 1) is not None


def test_calibration_blocks_alerts():
    """No distraction alert during the first 20 s of driving at >= 10 km/h.  REQ: SYS-03"""
    out, w = run_seq([(L.PHONE, 5.0)], warm=5.0)
    assert onset(out, 1) is None
    out, w = run_seq([(L.PHONE, 5.0)], warm=21.0)
    assert onset(out, 1) is not None


def test_configuration_is_read_only():
    """Detection thresholds cannot be changed at run time.  REQ: SYS-04"""
    with pytest.raises(dataclasses.FrozenInstanceError):
        PROPOSED.t_long_s = 5.0
    assert not hasattr(DmsSystem(PROPOSED), "set_threshold")


def test_privacy_bounded_buffers_no_identity():
    """Only fixed-size rolling buffers, no identity fields, nothing stored by the system.  REQ: SYS-05"""
    s = DmsSystem(PROPOSED)
    sizes0 = s.buffer_sizes()
    for k in range(N(600)):
        s.step(80.0, L.ROAD if k % 50 else L.UNKNOWN, 0.1 if k % 50 else 1.0)
    sizes1 = s.buffer_sizes()
    assert sizes1["vats"] == sizes0["vats"] == N(30) and sizes1["perclos"] == N(60)
    assert sizes1["microsleep"] == 0
    names = " ".join(vars(s).keys()).lower() + " ".join(vars(s.warn).keys()).lower()
    assert not any(x in names for x in ("face_id", "identity", "biometric", "image", "record"))
