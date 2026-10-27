"""Drowsiness (PERCLOS, microsleep, sleep, unresponsive) and sensor availability."""
import math

import pytest

from dms import labels as L
from dms.config import PROPOSED
from dms.detectors import DEGRADED, FAILED, NOMINAL, UNAVAILABLE, Availability, Drowsiness
from dms.system import DmsSystem
from dms.warning import URGENCY

N = PROPOSED.n


def test_perclos_value_and_learning_phase():
    """PERCLOS only after a full 60 s window; value = closed / valid frames.  REQ: DRW-01, DRW-04"""
    d = Drowsiness(PROPOSED)
    vals = []
    for k in range(N(70)):
        closed = (k % 10) < 2                      # 20 % closed
        vals.append(d.update(k, 1.0 if closed else 0.1, True)[0])
    assert all(v == -1.0 for v in vals[:N(60) - 1])
    assert vals[-1] == pytest.approx(0.2, abs=0.01)


def test_perclos_ignores_invalid_frames():
    """Frames without a valid eyelid value are excluded; < 50 % valid -> no PERCLOS.  REQ: DRW-01"""
    d = Drowsiness(PROPOSED)
    for k in range(N(60)):
        v = 1.0 if k % 4 == 0 else 0.1
        out = d.update(k, v if k % 2 == 0 else math.nan, True)
    assert out[0] == pytest.approx(0.5, abs=0.01)  # every valid frame alternates closed / open
    d2 = Drowsiness(PROPOSED)
    for k in range(N(60)):
        out = d2.update(k, 0.1 if k % 3 == 0 else math.nan, True)
    assert out[0] == -1.0 and out[2] == 0


def test_level_hysteresis_and_microsleep_escalation():
    """L2 at 0.20, kept until < 0.15; two microsleeps in 5 min also give L2.  REQ: DRW-02"""
    d = Drowsiness(PROPOSED)
    level = 0
    for k in range(N(60)):
        level = d.update(k, 1.0 if (k % 100) < 22 else 0.1, True)[2]
    assert level == 2
    for k in range(N(60), N(60) + N(20)):          # lower closure to ~0.17: stays at 2 (exit is 0.15)
        level = d.update(k, 1.0 if (k % 100) < 17 else 0.1, True)[2]
    assert level == 2
    d2 = Drowsiness(PROPOSED)
    k = 0
    for _ in range(2):
        for _ in range(N(1.3)):
            d2.update(k, 1.0, True); k += 1
        for _ in range(N(20)):
            out = d2.update(k, 0.1, True); k += 1
    assert out[2] == 2


def closed_eyes(cfg, seconds, speed=60.0):
    s = DmsSystem(cfg)
    for _ in range(cfg.n(30)):
        s.step(speed, L.ROAD, 0.1)
    return [s.step(speed, L.UNKNOWN, 1.0) for _ in range(cfg.n(seconds))]


def test_microsleep_sleep_unresponsive_frames():
    """Eye closure: microsleep at 1.2 s, sleep at 3.0 s, unresponsive + EF request at 6.0 s.  REQ: DRW-05, DRW-06, DRW-07"""
    outs = closed_eyes(PROPOSED, 7.0)
    src = [o[3] for o in outs]
    assert src.index("microsleep") + 1 == N(1.2)
    assert src.index("sleep") + 1 == N(3.0)
    assert src.index("unresponsive") + 1 == N(6.0) and outs[N(6.0) - 1][6]
    assert all(o[0] and o[1] and o[2] for o in outs[N(1.2) - 1:])          # visual + acoustic + haptic


def test_sleep_class_warnings_more_urgent():
    """Microsleep / sleep / unresponsive are distinct and more urgent than distraction and drowsiness.  REQ: DRW-08"""
    assert URGENCY["unresponsive"] > URGENCY["sleep"] > URGENCY["microsleep"] > URGENCY["distraction_long_glance"]
    assert URGENCY["distraction_vats"] >= URGENCY["drowsy"] > URGENCY["drowsy_info"]


def test_inform_after_5s_and_audible_once_per_journey():
    """Face lost -> information after 5 s (<= 10 s), audible only once per journey.  REQ: AVL-03"""
    a = Availability(PROPOSED)
    infos, pulses = [], []
    for k in range(N(12)):
        st, info, pulse = a.update(True, k < N(1) or k > N(11), True)
        infos.append(info); pulses.append(pulse)
    assert infos.index(True) == N(1) + N(5) - 1
    for k in range(N(10)):                         # second occlusion in the same journey: no second sound
        a.update(True, True, True)
    for k in range(N(8)):
        st, info, pulse = a.update(True, False, True)
        pulses.append(pulse)
    assert sum(pulses) == 1 and info


def test_failure_latched_while_present_and_system_keeps_running():
    """No light -> constant failure tell-tale after 1 s, held while present, cleared 2 s after; detection resumes.
    REQ: AVL-01, SYS-09"""
    s = DmsSystem(PROPOSED)
    for _ in range(N(30)):
        s.step(60.0, L.ROAD, 0.1)
    outs = [s.step(60.0, L.PHONE, 0.4, camera_ok=False) for _ in range(N(6))]
    assert outs[N(1.0) - 1][8] and outs[-1][8] and outs[-1][9] == FAILED
    assert not any(o[1] for o in outs)            # no distraction alert from untrusted data
    outs = [s.step(60.0, L.ROAD, 0.1) for _ in range(N(3))]
    assert not outs[-1][8] and outs[-1][9] == NOMINAL
    outs = [s.step(60.0, L.PHONE, 0.4) for _ in range(N(4))]
    assert any(o[1] for o in outs)                # system was never switched off


def test_degraded_mode_head_pose_only():
    """Sunglasses: gaze labels from head pose still work; eyelid metrics are not used.  REQ: AVL-04"""
    s = DmsSystem(PROPOSED)
    for _ in range(N(30)):
        s.step(60.0, L.ROAD, math.nan, eyes=False)
    outs = [s.step(60.0, L.INFOTAINMENT, math.nan, eyes=False) for _ in range(N(4))]
    assert outs[-1][9] == DEGRADED and any(o[1] for o in outs)
    outs = [s.step(60.0, L.UNKNOWN, 1.0, eyes=False) for _ in range(N(4))]
    assert not any(o[3] in ("microsleep", "sleep") for o in outs)
    a = Availability(PROPOSED)
    assert a.update(True, True, False)[0] == DEGRADED and a.update(True, False, True)[0] == UNAVAILABLE
