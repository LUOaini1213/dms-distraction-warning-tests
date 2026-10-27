"""Distraction logic: long glance, tolerance, speed independence, VATS, driving-task glances, ADAS coupling."""
import pytest

from dms import labels as L
from dms.config import ADDW_MIN, NAIVE, PROPOSED
from dms.detectors import LongGlance, Vats
from dms.geometry import ADDW_ZONES, ENCAP_LOCATIONS, addw_area, in_area3
from dms.system import DmsSystem

N = PROPOSED.n


def drive(cfg, seq, speed=55.0, warm=30.0):
    """seq = [(label, seconds), ...] after `warm` s of road gaze; returns list of step outputs."""
    s = DmsSystem(cfg)
    out = [s.step(speed, L.ROAD, 0.1) for _ in range(cfg.n(warm))]
    for lab, dur in seq:
        out += [s.step(speed, lab, 0.1) for _ in range(cfg.n(dur))]
    return out, cfg.n(warm)


def first(out, idx, pred=bool, start=0):
    for k in range(start, len(out)):
        if pred(out[k][idx]):
            return k
    return None


def test_area_classification():
    """ADDW Area 1/2/3 on the illustrative cabin.  REQ: DIS-01"""
    area3 = {k for k, z in ADDW_ZONES.items() if z.area == "3"}
    assert area3 == set("abcdefkln")
    assert ADDW_ZONES["i"].area == "2" and ADDW_ZONES["g"].area == "2"
    assert ADDW_ZONES["m"].area == "-" and ADDW_ZONES["j"].area == "-"     # between the windscreen margin and -30 deg
    assert addw_area(150, 0) == "1" and addw_area(0, 30) == "1"            # rear passenger, roof
    assert not in_area3(0, -29.9) and in_area3(0, -30.1)                    # 30 deg plane
    assert addw_area(*ENCAP_LOCATIONS["phone held in view of windscreen"][2:]) == "2"   # geometry would miss it


def test_long_glance_threshold_exact_frames():
    """Alert on the 99th frame of a non-driving glance (3.3 s at 30 Hz), visual pre-alert on the 60th.  REQ: DIS-02"""
    out, w = drive(PROPOSED, [(L.PHONE, 5.0)])
    assert first(out, 1) - w + 1 == N(3.3) == 99
    assert first(out, 0) - w + 1 == N(2.0) == 60
    out, w = drive(PROPOSED, [(L.PHONE, 98 * PROPOSED.dt), (L.ROAD, 2.0)])
    assert first(out, 1) is None


def test_presets_thresholds():
    """ADDW_MIN uses 3.5 s at >= 50 km/h and 6.0 s below; PROPOSED one 3.3 s threshold.  REQ: DIS-02"""
    for v, n_expected in ((55.0, ADDW_MIN.n(3.5)), (30.0, ADDW_MIN.n(6.0))):
        out, w = drive(ADDW_MIN, [(L.PHONE, 7.0)], speed=v)
        assert first(out, 1) - w + 1 == n_expected
    out, w = drive(PROPOSED, [(L.PHONE, 7.0)], speed=30.0)
    assert first(out, 1) - w + 1 == N(3.3)


def test_gap_tolerance_counts_and_resets():
    """<= 0.3 s interruptions are counted, longer ones reset.  REQ: DIS-04"""
    lg = LongGlance(L.NON_DRIVING, False, N(0.3))
    for _ in range(30):
        lg.update(L.PHONE)
    for _ in range(9):
        lg.update(L.ROAD)
    assert lg.update(L.PHONE) == 30 + 9 + 1
    for _ in range(10):
        lg.update(L.ROAD)
    assert lg.n == 0 and not lg.active


def test_unknown_freezes_not_resets():
    """'unknown' beyond the tolerance freezes the glance timer; it does not reset it.  REQ: DIS-04"""
    lg = LongGlance(L.NON_DRIVING, False, N(0.3))
    for _ in range(30):
        lg.update(L.PHONE)
    for _ in range(60):
        lg.update(L.UNKNOWN)
    assert lg.n == 30 and lg.active
    assert lg.update(L.PHONE) == 31                       # gap longer than tolerance is not added


def test_timer_runs_below_activation_speed():
    """Timer counts at 15 km/h; the alert comes as soon as the speed reaches 20 km/h.  REQ: DIS-03, SYS-01"""
    s = DmsSystem(PROPOSED)
    for _ in range(N(30)):
        s.step(15.0, L.ROAD, 0.1)
    outs = [s.step(15.0, L.PHONE, 0.1) for _ in range(N(4.0))]
    assert not any(o[1] for o in outs) and outs[-1][10] == pytest.approx(4.0)
    o = s.step(20.0, L.PHONE, 0.1)
    assert o[1] and o[3] == "distraction_long_glance"


def test_vats_window_and_reset():
    """Sliding 30 s sum; a 2.0 s road run clears the episode; mirror does not add.  REQ: DIS-05"""
    v = Vats(L.NON_DRIVING, False, N(30.0), N(2.0))
    for _ in range(5):
        for _ in range(N(1.8)):
            v.update(L.INFOTAINMENT)
        for _ in range(N(0.9)):
            v.update(L.ROAD)
    assert v.sum == 5 * N(1.8)
    for _ in range(N(1.0)):
        v.update(L.MIRROR)
    assert v.sum == 5 * N(1.8) and v.road_run == 0
    for _ in range(N(2.0)):
        v.update(L.ROAD)
    assert v.sum == 0
    w = Vats(L.NON_DRIVING, False, N(30.0), N(2.0))      # window: 25 % duty never reaches 10 s
    for _ in range(40):
        for _ in range(N(0.5)):
            w.update(L.PHONE)
        for _ in range(N(1.5)):
            w.update(L.ROAD)
    assert w.sum <= N(7.5) + 1


def test_phone_counts_anywhere():
    """phone label counts for long glance and VATS regardless of where the phone is.  REQ: DIS-06"""
    out, w = drive(PROPOSED, [(L.PHONE, 4.0)])
    assert out[first(out, 1)][3] == "distraction_long_glance"
    seq = [(L.PHONE, 1.5), (L.ROAD, 0.8)] * 8
    out, w = drive(PROPOSED, seq)
    k = first(out, 1)
    assert out[k][3] == "distraction_vats" and out[k][11] >= 10.0


def test_driving_task_glances_not_warned():
    """Mirror / instrument glances never trigger a distraction alert in PROPOSED (but do in NAIVE).  REQ: DIS-07"""
    for lab in (L.MIRROR, L.INSTRUMENT):
        out, w = drive(PROPOSED, [(lab, 6.0)])
        assert first(out, 1) is None
    out, w = drive(NAIVE, [(L.MIRROR, 6.0)])
    assert first(out, 1) is not None


def test_fss_on_and_off():
    """ADAS sensitivity: on after 0.8 s continuous gaze away, off after >= 1 s continuous road gaze; held 2 s after
    an alert ends.  REQ: DIS-08"""
    out, w = drive(PROPOSED, [(L.MIRROR, 1.5), (L.ROAD, 3.0)])
    on = first(out, 5)
    assert on - w + 1 == N(0.8)
    off = first(out, 5, lambda x: not x, start=on)
    assert off - (w + N(1.5)) + 1 == N(1.0)            # off on the 30th consecutive road frame
    out, w = drive(PROPOSED, [(L.PHONE, 5.0), (L.ROAD, 5.0)])
    alert_end = max(k for k, o in enumerate(out) if o[1])
    off = first(out, 5, lambda x: not x, start=alert_end)
    assert (off - alert_end) * PROPOSED.dt >= PROPOSED.t_fss_off_state_s - 1e-9
