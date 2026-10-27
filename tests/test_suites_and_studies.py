"""Scenario suites (regulation, boundary), NHTSA criteria and the DDAW Part 2 computation."""
import math

import numpy as np
import pandas as pd
import pytest

from dms import ddaw, nhtsa
from dms import labels as L
from dms import regulations as R
from dms.config import ADDW_MIN, PROPOSED
from dms.suites import addw_point_verdict, boundary_cases, evaluate, evaluate_boundary, regulation_cases


@pytest.fixture(scope="module")
def proposed_rows():
    return pd.DataFrame([evaluate(c, PROPOSED) for c in regulation_cases(PROPOSED)])


def test_regulation_cases_proposed(proposed_rows):
    """All PROPOSED regulation / protocol cases pass; ADDW zones outside Area 3 are 'not applicable'.
    REQ: TST-01, TST-02, TST-04"""
    d = proposed_rows
    assert set(d.verdict) <= {"通过", "不适用"}
    addw = d[d.suite == "ADDW"]
    assert len(addw) == 28 and (addw[addw.required == "Y"].verdict == "通过").all()
    assert (addw[addw.required == "N"].verdict == "不适用").all()
    assert (d[d.suite != "ADDW"].verdict == "通过").all()
    assert (addw[addw.required == "Y"].t_acoustic_s <= R.ADDW_TEST_FN_HIGH_S).all()


def test_addw_minimum_misses_euro_ncap():
    """The bare ADDW minimum fails Euro NCAP long distraction below 50 km/h and has no VATS.  REQ: DIS-02, DIS-05"""
    d = pd.DataFrame([evaluate(c, ADDW_MIN) for c in regulation_cases(ADDW_MIN)])
    assert (d[(d.suite == "ADDW") & (d.required == "Y")].verdict == "通过").all()
    ld = d[d.test_id.str.startswith("ENCAP-LD-nondriving")]
    assert (ld[ld.speed_kph < 50].verdict == "不通过").all() and (ld[ld.speed_kph >= 50].verdict == "通过").all()
    assert (d[d.test_id.str.startswith(("ENCAP-VATS-nondriving", "ENCAP-PHONE"))].verdict == "不通过").all()


def test_addw_retest_logic():
    """Part 2 4-6: a false negative is re-tested up to twice; 'fail' only if both re-tests miss.  REQ: TST-01"""
    assert addw_point_verdict([False]) == "pass"
    assert addw_point_verdict([True, False]) == "pass"
    assert addw_point_verdict([True, True, False]) == "pass"
    assert addw_point_verdict([True, True, True]) == "fail"
    with pytest.raises(ValueError):
        addw_point_verdict([True, True])


def test_boundary_cases():
    """Every boundary case behaves as designed (just below / just above each threshold).
    REQ: SYS-01, DIS-02, DIS-03, DIS-04, DIS-05, DIS-08, DIS-09, DRW-02, DRW-03, DRW-05, DRW-06, DRW-07, AVL-03"""
    d = pd.DataFrame([evaluate_boundary(b) for b in boundary_cases()])
    assert (d.verdict == "通过").all(), d[d.verdict != "通过"][["test_id", "observed_event"]].to_string()
    lat = d[d.test_id.str.startswith("BC-LAT")].set_index("test_id")
    assert "NCAP 满足=是" in lat.loc["BC-LAT-0.3", "observed_event"] and "NCAP 满足=否" in lat.loc["BC-LAT-0.5", "observed_event"]
    assert "ADDW 满足=是" in lat.loc["BC-LAT-0.7", "observed_event"] and "ADDW 满足=否" in lat.loc["BC-LAT-0.8", "observed_event"]


def test_nhtsa_criteria_rounding_and_rule():
    """VI.E.14: 15 % rounded up, mean <= 2.0 s, total <= 12.0 s; >= 21 of 24 for each criterion.  REQ: TST-05"""
    seq = [(L.INFOTAINMENT, d) for d in (2.1, 2.2, 1.0, 1.0, 1.0, 1.0, 1.0)]       # 7 glances, 2 > 2 s
    m = nhtsa.glance_metrics(seq)
    assert m["allowed_gt2s"] == 2 and m["crit_a"] and m["crit_b"] and m["crit_c"]
    m = nhtsa.glance_metrics(seq + [(L.INFOTAINMENT, 2.5)])                          # 8 glances, 3 > 2 s
    assert m["allowed_gt2s"] == 2 and not m["crit_a"]
    m = nhtsa.glance_metrics([(L.INFOTAINMENT, 1.9)] * 7)                          # 13.3 s total
    assert m["crit_b"] and not m["crit_c"]
    rows = pd.DataFrame({"crit_a": [True] * 21 + [False] * 3, "crit_b": [True] * 24, "crit_c": [True] * 20 + [False] * 4})
    acc = nhtsa.acceptance(rows)
    assert acc["pass_a"] and acc["pass_b"] and not acc["pass_c"] and not acc["task_acceptable"]


def test_nhtsa_participant_mix():
    """VI.A.4 / VI.A.5: 24 participants, six per age group, three men and three women in each.  REQ: TST-05"""
    ps = nhtsa.participants()
    assert len(ps) == R.NHTSA_PARTICIPANTS
    for g in R.NHTSA_AGE_GROUPS:
        grp = [p for p in ps if p.age_group == g]
        assert len(grp) == 6 and sum(p.sex == "M" for p in grp) == 3


def test_nhtsa_hmi_tradeoff():
    """Synthetic model: touch address entry fails (TEORT), voice entry and a stored destination pass; the DMS
    would warn during touch entry.  REQ: HMT-01, HMT-02"""
    per, summ = nhtsa.evaluate()
    s = summ.set_index("variant")
    assert not s.loc["touch_address", "task_acceptable"] and not s.loc["touch_address", "pass_c"]
    assert s.loc["voice_address", "task_acceptable"] and s.loc["touch_favourite", "task_acceptable"]
    assert s.loc["touch_address", "share_with_dms_alert"] > 0.5 and s.loc["voice_address", "share_with_dms_alert"] == 0


def test_ddaw_event_classification():
    """Part 2 5.1.4 / 5.1.5 TP / FN / outlier / exclusion rules.  REQ: DRW-10"""
    t = [0, 300, 600, 900, 1200]
    assert ddaw.classify(t, [5, 6, 7, 8, 8], [650])["tp"] == 1                 # warning with next rating 7
    assert ddaw.classify(t, [5, 6, 8, 8, 9], [])["fn"] >= 1                     # 6 -> 8 missed, next 8
    r = ddaw.classify(t, [5, 6, 8, 7, 7], [])
    assert r["fn"] == 0 and r["tn_outlier"] == 1                                # 6 -> 8 -> 7
    assert ddaw.classify(t, [5, 6, 8, 6, 6], [])["excluded"]                    # 6 -> 8 -> 6 unreliable
    assert ddaw.classify(t, [3, 4, 5, 5, 5], [400])["fp"] == 1


def test_ddaw_acceptance_formula():
    """Part 2 8.1: mean > 40 % or mean - 1.645 * SD > 20 %, with the open-road / long-interval adjustments.  REQ: DRW-10"""
    s = [1.0, 0.5, 0.5, 0.0, 1.0, 0.5, 0.5, 0.5, 1.0, 0.0]
    a = ddaw.acceptance(s)
    mean = float(np.mean(s))
    sd = math.sqrt(sum((x - mean) ** 2 for x in s) / len(s))
    assert a["mean_sensitivity"] == pytest.approx(mean, abs=1e-4) and a["sd"] == pytest.approx(sd, abs=1e-4)
    assert a["lower_bound"] == pytest.approx(mean - 1.645 * sd, abs=1e-4)
    assert a["crit_a"] and not a["crit_b"] and a["effective"]
    assert ddaw.acceptance([0.38] * 10, open_road=True)["effective"]             # 35 % on an open road
    assert not ddaw.acceptance([0.42] * 10, interval_over_15min=True)["crit_a"]   # 45 % needed
