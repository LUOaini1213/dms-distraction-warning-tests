"""Scenario test suites (SYNTHETIC inputs): regulation / protocol test cases and boundary cases.

Regulation cases follow the procedures where the documents specify them:
  ADDW Annex I Part 2      fixation zones (a)-(n) [1.4.2], one speed in 20-35 and one in 50-65 km/h [1.5.1],
                           60 s attentive before [2.3.1], hold until the expected time + 3 s [2.3.8],
                           false negative if no warning within 4.0 s / 6.5 s [3.1 / 3.2], re-test <= 2 [4-5]
  Euro NCAP SD-202 v1.2    T0 = T_away - 4 s on-road [1.1], 20-80 km/h [4.1], hold <= 5 s and tolerance [0.2] s
                           for long distraction [4.1.1.1], VATS warning at declared dT_VATS +/- [2.0] s with
                           on-road gazes shorter than T_VATS_reset [4.1.1.2], eyes closed 4 s / 7 s for
                           microsleep / sleep [4.1.2], 13 s / 14 s unresponsive [4.2.3.3], 10 s hand occlusion [4.3]
  IVASA ICI-TP             5 s gaze at three points, 5 s eye closure, three repetitions [4.1.1 / 4.1.2]
Times in the result rows are measured from the TRUE start of the behaviour (T_away / T_close); the DMS sees
the inputs 0.1 s late (perception latency, DA-02) and decides at the end of a frame (t_k + dt).
The pass window of the SD-202 long-distraction tolerance is drawn in a diagram that could not be read; it is
evaluated here as |T_warn - (T_away + declared dT_LGA)| <= 0.2 s (assumption, see report section 3).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import labels as L
from . import regulations as R
from .config import PROPOSED, DmsConfig
from .geometry import ADDW_ZONES, ENCAP_LOCATIONS
from .scenarios import DATA_ORIGIN, Seg, timeline
from .system import first_onset, run

LEAD = 60.0            # ADDW Part 2 2.3.1 (60 s) and SD-202 3.2.1 (>= 1 min attentive)
LATENCY = 0.1          # DA-02
PASS, FAIL, NA = "通过", "不通过", "不适用"


@dataclass
class Case:
    test_id: str
    suite: str
    source: str
    clause: str
    description: str
    build: Callable[[], pd.DataFrame | list]      # one journey (DataFrame) or several journeys (list)
    t_ref: float
    check: Callable[["Case", object], tuple[str, dict]]
    meta: dict = field(default_factory=dict)


def _rel(t: float, ref: float) -> float:
    return round(t - ref, 3) if not math.isnan(t) else float("nan")


def evaluate(case: Case, cfg: DmsConfig = PROPOSED, latency_s: float = LATENCY) -> dict:
    scn = case.build()
    out = [run(s, cfg, latency_s) for s in scn] if isinstance(scn, list) else run(scn, cfg, latency_s)
    verdict, m = case.check(case, out)
    o = out[0] if isinstance(out, list) else out
    meta = {k: case.meta.get(k, "-") for k in ("speed_kph", "gaze_label", "location", "movement", "addw_area", "required")}
    row = {"test_id": case.test_id, "suite": case.suite, "config": cfg.name, "source": case.source,
           "clause": case.clause, "description": case.description, **meta,
           "t_visual_s": _rel(first_onset(o, "visual", case.t_ref - 0.5), case.t_ref),
           "t_acoustic_s": _rel(first_onset(o, "acoustic", case.t_ref - 0.5), case.t_ref),
           "t_haptic_s": _rel(first_onset(o, "haptic", case.t_ref - 0.5), case.t_ref),
           **m, "verdict": verdict, "data_origin": DATA_ORIGIN}
    return row


# ============================================================================ check helpers
def _t_alert(out, ref: float) -> float:
    """Distraction warning start = acoustic onset of a distraction alert (ADDW 3.4.1.1)."""
    return first_onset(out, "acoustic", ref - 0.5, "distraction")


def addw_point_verdict(false_negatives: list[bool]) -> str:
    """ADDW Annex I Part 2 4.1 / 5.1 / 5.2 for one fixation point.  false_negatives = [first test, re-test 1,
    re-test 2]; re-tests exist only after a false negative.  'fail' only if both re-tests are false negatives;
    a true positive (or 'not applicable') in a re-test clears the point.  6.1: one 'fail' fails the spot check."""
    if not false_negatives or not false_negatives[0]:
        return "pass"
    retests = false_negatives[1:1 + R.ADDW_TEST_RETEST_MAX]
    if len(retests) < R.ADDW_TEST_RETEST_MAX:
        if retests and not retests[-1]:
            return "pass"
        raise ValueError("a false-negative re-test must be followed by the next re-test (5.1 / 5.2)")
    return "fail" if all(retests) else "pass"


def chk_within(limit: float, required: bool = True):
    def f(c: Case, out):
        t = _t_alert(out, c.t_ref) - c.t_ref
        m = {"criterion": f"告警 ≤ {limit:g} s", "measured_s": round(t, 3)}
        if not required:
            return NA, {**m, "note": "该区域不在 Area 3（几何假设），ADDW 不要求报警；记录本设计的行为"}
        fn = [not (t <= limit + 1e-9)]
        if fn[0] and c.suite == "ADDW":
            # re-tests with a different distracted behaviour (Part 2 4.2): repeat with shifted blink phase.
            # The synthetic model is deterministic, so a re-test can only confirm the first result.
            fn += [fn[0], fn[0]]
        verdict = addw_point_verdict(fn) if c.suite == "ADDW" else ("pass" if not fn[0] else "fail")
        return (PASS if verdict == "pass" else FAIL), m
    return f


def chk_window(declared: float, tol: float):
    def f(c: Case, out):
        t = _t_alert(out, c.t_ref) - c.t_ref
        ok = not math.isnan(t) and abs(t - declared) <= tol + 1e-9
        return (PASS if ok else FAIL), {"criterion": f"{declared:g} ± {tol:g} s", "measured_s": round(t, 3)}
    return f


def chk_vats(t_expected_rel: float, tol: float):
    def f(c: Case, out):
        t = _t_alert(out, c.t_ref) - c.t_ref
        ok = not math.isnan(t) and abs(t - t_expected_rel) <= tol + 1e-9
        return (PASS if ok else FAIL), {"criterion": f"{t_expected_rel:g} ± {tol:g} s", "measured_s": round(t, 3)}
    return f


def chk_driving_task(fss_limit: float = R.ENCAP_FSS_LATENCY_S, allow_warning: bool = False):
    def f(c: Case, out):
        t_fss = first_onset(out, "fss", c.t_ref - 0.5) - c.t_ref
        t_w = _t_alert(out, c.t_ref) - c.t_ref
        ok = t_fss <= fss_limit + 1e-9 and (allow_warning or math.isnan(t_w))
        note = "规程对驾驶相关注视只给干预分（DE §1.4 表）：只要求 ADAS 灵敏度联动"
        if not math.isnan(t_w):
            note += f"；产生了额外报警 {t_w:.2f} s（侧窗按 off_road 计）"
        return (PASS if ok else FAIL), {"criterion": f"FSS ≤ {fss_limit:g} s，不要求报警", "measured_s": round(t_fss, 3),
                                        "note": note}
    return f


def chk_source(source: str, declared: float, tol: float):
    def f(c: Case, out):
        t = first_onset(out, "acoustic", c.t_ref - 0.5, source) - c.t_ref
        ok = not math.isnan(t) and abs(t - declared) <= tol + 1e-9
        return (PASS if ok else FAIL), {"criterion": f"{source} {declared:g} ± {tol:g} s", "measured_s": round(t, 3)}
    return f


def chk_ef(lo: float, hi: float):
    def f(c: Case, out):
        t = first_onset(out, "ef_request", c.t_ref - 0.5) - c.t_ref
        ok = not math.isnan(t) and lo - 1e-9 <= t <= hi + 1e-9
        return (PASS if ok else FAIL), {"criterion": f"无响应判定 {lo:g}–{hi:g} s", "measured_s": round(t, 3),
                                        "note": "EF（紧急停车功能）本身不在本项目范围，只检查判定标志"}
    return f


def chk_info(limit: float):
    def f(c: Case, out):
        t = first_onset(out, "info", c.t_ref - 0.5) - c.t_ref
        return (PASS if t <= limit + 1e-9 else FAIL), {"criterion": f"提示 ≤ {limit:g} s", "measured_s": round(t, 3)}
    return f


# ============================================================================ builders
def _glance(speed: float, label: str, hold: float, post: float = 15.0, eyes: bool = True, pre: float = LEAD):
    return lambda: timeline([Seg(pre, speed=speed, eyes=eyes), Seg(hold, label, eyes=eyes), Seg(post, eyes=eyes)], speed)


def _pattern(speed: float, labels: list[str], off: float, on: float, cycles: int, eyes: bool = True):
    def b():
        segs = [Seg(LEAD, speed=speed, eyes=eyes)]
        for i in range(cycles):
            segs += [Seg(off, labels[i % len(labels)], eyes=eyes), Seg(on, eyes=eyes, blinks=False)]
        segs.append(Seg(15.0, eyes=eyes))
        return timeline(segs, speed)
    return b


def _closed(speed: float, hold: float, eyes: bool = True):
    return lambda: timeline([Seg(LEAD, speed=speed, eyes=eyes), Seg(hold, L.UNKNOWN, eyelid=1.0, eyes=eyes),
                             Seg(15.0, eyes=eyes)], speed)


def _vats_expected(off: float, on: float, target: float) -> float:
    """Time from the first T_away until the cumulative off-road time reaches `target`."""
    full = int(target // off)
    rest = target - full * off
    if rest < 1e-9:
        return full * off + (full - 1) * on
    return full * (off + on) + rest


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in s.lower()).strip("-").replace("--", "-")


def regulation_cases(cfg: DmsConfig = PROPOSED, seed: int = 7) -> list[Case]:
    rng = np.random.default_rng(seed)
    cases: list[Case] = []
    # ------------------------------------------------------------------ ADDW Annex I Part 2
    for z in ADDW_ZONES.values():
        required = z.area == "3"
        for v, (limit, expected) in ((25.0, (R.ADDW_TEST_FN_LOW_S, R.ADDW_T_MAX_LOW_S)),
                                     (55.0, (R.ADDW_TEST_FN_HIGH_S, R.ADDW_T_MAX_HIGH_S))):
            hold = expected + R.ADDW_TEST_HOLD_EXTRA_S
            cases.append(Case(
                f"ADDW-P2-{z.key}-{int(v)}", "ADDW", R.ADDW_SRC,
                "Annex I Part 2 §1.4.2, §1.5.1, §2.3.1, §2.3.8, " + ("§3.2" if v < 50 else "§3.1"),
                f"注视点 ({z.key}) {z.name}，{v:g} km/h，保持 {hold:g} s", _glance(v, z.label, hold), LEAD,
                chk_within(limit, required),
                {"speed_kph": v, "gaze_label": z.label, "location": z.name, "movement": "-",
                 "addw_area": z.area, "required": "Y" if required else "N"}))
    # ------------------------------------------------------------------ Euro NCAP long distraction
    ld = {("non_driving", "owl"): ["driver side window", "passenger side window", "passenger footwell", "passenger face",
                                   "in-vehicle infotainment system"],
          ("non_driving", "lizard"): ["in-vehicle infotainment system", "glovebox"],
          ("non_driving", "body_lean"): ["passenger footwell", "rear passenger"],
          ("driving", "owl"): ["rear view mirror", "passenger side mirror", "driver side mirror"],
          ("driving", "lizard"): ["instrument cluster", "driver side mirror", "rear view mirror"]}
    for (task, mov), locs in ld.items():
        for loc in locs:
            v = float(rng.integers(20, 81))
            lab = ENCAP_LOCATIONS[loc][0]
            chk = chk_window(cfg.declared_lga_s, R.ENCAP_LD_TOL_S) if task == "non_driving" else chk_driving_task()
            cases.append(Case(
                f"ENCAP-LD-{task.replace('_', '')}-{mov}-{_slug(loc)}", "ENCAP", R.ENCAP_SD202_SRC,
                "DE §1.2.1, §1.4.1; SD-202 §1.1, §4.1, §4.1.1.1" + ("" if task == "non_driving" else "; DE §1.4.3.1"),
                f"长时分心 {mov} → {loc}，{v:g} km/h，注视 5 s", _glance(v, lab, R.ENCAP_LD_HOLD_MAX_S), LEAD, chk,
                {"speed_kph": v, "gaze_label": lab, "location": loc, "movement": mov, "addw_area": "-",
                 "required": "Y"}))
    # ------------------------------------------------------------------ Euro NCAP VATS
    vats = {("non_driving", "owl"): ["in-vehicle infotainment system", "passenger footwell"],
            ("non_driving", "lizard"): ["passenger footwell", "in-vehicle infotainment system"],
            ("driving", "owl"): ["rear view mirror", "passenger side mirror", "driver side mirror", "passenger side window"],
            ("driving", "lizard"): ["instrument cluster", "driver side mirror", "rear view mirror", "driver side window"]}
    off, on, cyc = 1.8, 0.9, 8
    t_exp = _vats_expected(off, on, cfg.declared_vats_s)
    for (task, mov), locs in vats.items():
        for loc in locs:
            v = float(rng.integers(20, 81))
            lab = ENCAP_LOCATIONS[loc][0]
            chk = chk_vats(t_exp, R.ENCAP_VATS_TOL_S) if task == "non_driving" else chk_driving_task(allow_warning=True)
            cases.append(Case(
                f"ENCAP-VATS-{task.replace('_', '')}-{mov}-{_slug(loc)}", "ENCAP", R.ENCAP_SD202_SRC,
                "DE §1.2.2; SD-202 §4.1.1.2" + ("" if task == "non_driving" else "; DE §1.4.3.1"),
                f"VATS {mov} → {loc}：{off:g} s 离开 / {on:g} s 路面 × {cyc}，{v:g} km/h", _pattern(v, [lab], off, on, cyc),
                LEAD, chk, {"speed_kph": v, "gaze_label": lab, "location": loc, "movement": mov, "addw_area": "-",
                            "required": "Y"}))
    v = float(rng.integers(20, 81))
    cases.append(Case(
        "ENCAP-VATS-multi-lizard", "ENCAP", R.ENCAP_SD202_SRC, "DE §1.2.2 (Multi-location); SD-202 §4.1.1.2",
        f"多位置 VATS：中控屏 / 副驾脚坑交替，{v:g} km/h", _pattern(v, [L.INFOTAINMENT, L.OFF_ROAD], off, on, cyc), LEAD,
        chk_vats(t_exp, R.ENCAP_VATS_TOL_S),
        {"speed_kph": v, "gaze_label": "infotainment+off_road", "location": "multi", "movement": "lizard",
         "addw_area": "-", "required": "Y"}))
    # ------------------------------------------------------------------ Euro NCAP phone use
    p_off, p_on, p_cyc = 1.5, 0.8, 9
    t_exp_p = _vats_expected(p_off, p_on, cfg.declared_vats_s)
    phone = {("basic", "owl"): ["driver knee outboard", "driver knee inboard", "driver lap",
                                "phone mounted on dashboard outboard", "phone in charge port"],
             ("basic", "lizard"): ["driver knee outboard", "driver knee inboard", "driver lap",
                                   "phone held centre of steering wheel", "phone in charge port"],
             ("advanced", "lizard"): ["phone mounted on dashboard outboard", "phone held 9-11 / 13-15 o'clock on wheel",
                                      "phone held in view of windscreen", "phone held in view of instrument cluster"]}
    for (kind, mov), locs in phone.items():
        for loc in locs:
            v = float(rng.integers(20, 81))
            cases.append(Case(
                f"ENCAP-PHONE-{kind}-{mov}-{_slug(loc)}", "ENCAP", R.ENCAP_SD202_SRC, "DE §1.2.3; SD-202 §4.1.1.2",
                f"手机使用（{kind}）{mov} → {loc}：{p_off:g}/{p_on:g} s × {p_cyc}，{v:g} km/h",
                _pattern(v, [L.PHONE], p_off, p_on, p_cyc), LEAD, chk_vats(t_exp_p, R.ENCAP_VATS_TOL_S),
                {"speed_kph": v, "gaze_label": L.PHONE, "location": loc, "movement": mov, "addw_area": "-",
                 "required": "Y"}))
    # ------------------------------------------------------------------ Euro NCAP non-transient states
    v = float(rng.integers(20, 81))
    cases += [
        Case("ENCAP-MS", "ENCAP", R.ENCAP_SD202_SRC, "DE §1.3.3, §1.4.2; SD-202 §1.1, §4.1.2.2",
             f"微睡眠：闭眼 4 s，{v:g} km/h", _closed(v, R.ENCAP_MS_HOLD_MAX_S), LEAD,
             chk_source("microsleep", cfg.declared_microsleep_s, R.ENCAP_LD_TOL_S),
             {"speed_kph": v, "gaze_label": "unknown (eyes closed)", "location": "-", "movement": "-", "addw_area": "-",
              "required": "Y"}),
        Case("ENCAP-SLEEP", "ENCAP", R.ENCAP_SD202_SRC, "DE §1.3.4, §1.4.2; SD-202 §1.1, §4.1.2.3",
             f"睡眠：闭眼 7 s，{v:g} km/h", _closed(v, R.ENCAP_SLEEP_HOLD_MAX_S), LEAD,
             chk_source("sleep", R.ENCAP_SLEEP_S, R.ENCAP_LD_TOL_S),
             {"speed_kph": v, "gaze_label": "unknown (eyes closed)", "location": "-", "movement": "-", "addw_area": "-",
              "required": "Y"}),
        Case("ENCAP-UNRESP-sleep", "ENCAP", R.ENCAP_SD202_SRC, "DE §1.3.5; SD-202 §4.2.3.3.1",
             f"无响应（睡眠选项）：闭眼 13 s，{v:g} km/h", _closed(v, R.ENCAP_UNRESP_SLEEP_HOLD_S), LEAD,
             chk_ef(R.ENCAP_UNRESP_EYES_S, R.ENCAP_UNRESP_EYES_S + R.ENCAP_LD_TOL_S + LATENCY),
             {"speed_kph": v, "gaze_label": "unknown (eyes closed)", "location": "-", "movement": "-", "addw_area": "-",
              "required": "Y"}),
        Case("ENCAP-UNRESP-lap", "ENCAP", R.ENCAP_SD202_SRC, "DE §1.3.5; SD-202 §4.2.3.3.2",
             f"无响应（分心选项）：注视腿部 14 s，{v:g} km/h", _glance(v, L.OFF_ROAD, R.ENCAP_UNRESP_LAP_HOLD_S), LEAD,
             chk_ef(cfg.declared_lga_s + R.ENCAP_UNRESP_GAZE_S - R.ENCAP_LD_TOL_S,
                    cfg.declared_lga_s + R.ENCAP_UNRESP_GAZE_S + R.ENCAP_LD_TOL_S),
             {"speed_kph": v, "gaze_label": L.OFF_ROAD, "location": "driver lap", "movement": "owl", "addw_area": "3",
              "required": "Y"}),
        Case("ENCAP-NF-hand", "ENCAP", R.ENCAP_SD202_SRC, "DE §1.1.1.2 (Inform); SD-202 §4.3",
             "不可用：手握方向盘上部遮挡仪表处摄像头 ≥10 s",
             lambda: timeline([Seg(LEAD, speed=50.0), Seg(12.0, L.UNKNOWN, face=False), Seg(10.0)], 50.0), LEAD,
             chk_info(R.ENCAP_INFORM_S),
             {"speed_kph": 50.0, "gaze_label": "unknown", "location": "hand on wheel", "movement": "-", "addw_area": "-",
              "required": "Y"}),
    ]
    # ------------------------------------------------------------------ occlusion accessory (sunglasses < 15 %)
    def sg_ld(mov):
        lab_seen = L.INFOTAINMENT if mov == "owl" else L.ROAD     # head pose only: lizard eye movement invisible
        return lambda: timeline([Seg(LEAD, speed=60.0, eyes=False), Seg(5.0, lab_seen, eyes=False), Seg(15.0, eyes=False)], 60.0)

    def chk_sg_lizard(c, out):
        t_info = first_onset(out, "info", -1.0)
        t_w = _t_alert(out, c.t_ref) - c.t_ref
        ok = t_info <= R.ENCAP_INFORM_S + 1e-9 and math.isnan(t_w)
        return (PASS if ok else FAIL), {"criterion": "不能检测时 10 s 内提示", "measured_s": round(t_info, 3),
                                        "note": "降级：眼动（lizard）不可见，未报警；行程开始即已提示 DMS 受限"}

    def chk_sg_ms(c, out):
        t_info = first_onset(out, "info", -1.0)
        t_ms = first_onset(out, "acoustic", c.t_ref - 0.5, "microsleep")
        ok = t_info <= R.ENCAP_INFORM_S + 1e-9 and math.isnan(t_ms)
        return (PASS if ok else FAIL), {"criterion": "不能检测时 10 s 内提示", "measured_s": round(t_info, 3),
                                        "note": "眼睑不可见，微睡眠不可检测；已提示受限"}
    cases += [
        Case("ENCAP-OCC-sunglasses-inform", "ENCAP", R.ENCAP_DE_SRC, "DE §1.1.1.2; SD-202 §4, App. A",
             "墨镜（透过率<15%）从行程开始佩戴：提示", sg_ld("owl"), 0.0, chk_info(R.ENCAP_INFORM_S),
             {"speed_kph": 60.0, "gaze_label": "-", "location": "-", "movement": "-", "addw_area": "-", "required": "Y"}),
        Case("ENCAP-OCC-sunglasses-LD-owl", "ENCAP", R.ENCAP_DE_SRC, "DE 定义 Degraded system; SD-202 §4",
             "墨镜 + 长时分心 owl → 中控屏（头部姿态可见）", sg_ld("owl"), LEAD,
             chk_window(cfg.declared_lga_s, R.ENCAP_LD_TOL_S),
             {"speed_kph": 60.0, "gaze_label": L.INFOTAINMENT, "location": "in-vehicle infotainment system",
              "movement": "owl", "addw_area": "-", "required": "Y"}),
        Case("ENCAP-OCC-sunglasses-LD-lizard", "ENCAP", R.ENCAP_DE_SRC, "DE §1.1.1.2 (Inform if not functional)",
             "墨镜 + 长时分心 lizard → 中控屏（只动眼睛）", sg_ld("lizard"), LEAD, chk_sg_lizard,
             {"speed_kph": 60.0, "gaze_label": "road (head forward)", "location": "in-vehicle infotainment system",
              "movement": "lizard", "addw_area": "-", "required": "Y"}),
        Case("ENCAP-OCC-sunglasses-MS", "ENCAP", R.ENCAP_DE_SRC, "DE §1.1.1.2 (Inform if not functional)",
             "墨镜 + 闭眼 4 s", _closed(60.0, R.ENCAP_MS_HOLD_MAX_S, eyes=False), LEAD, chk_sg_ms,
             {"speed_kph": 60.0, "gaze_label": "unknown", "location": "-", "movement": "-", "addw_area": "-",
              "required": "Y"}),
    ]
    # ------------------------------------------------------------------ IVASA (beyond rating), 3 repetitions
    for pt, lab in zip(R.IVASA_DISTRACTION_POINTS, (L.INFOTAINMENT, L.OFF_ROAD, L.PHONE)):
        for rep in range(1, R.IVASA_REPEATS + 1):
            ph = 0.9 * rep
            cases.append(Case(
                f"IVASA-DIS-{_slug(pt)}-r{rep}", "IVASA", R.IVASA_SRC, "ICI-TP §4.1.1",
                f"IVASA 分心：注视 {pt} {R.IVASA_DISTRACTION_HOLD_S:g}+2 s，30 km/h，第 {rep} 次",
                (lambda lab=lab, ph=ph: timeline([Seg(LEAD, speed=30.0), Seg(R.IVASA_DISTRACTION_HOLD_S + R.IVASA_EXTRA_HOLD_S, lab),
                                                  Seg(15.0)], 30.0, phase_s=ph)),
                LEAD, chk_within(R.IVASA_DISTRACTION_HOLD_S),
                {"speed_kph": 30.0, "gaze_label": lab, "location": pt, "movement": "-", "addw_area": "-", "required": "Y"}))
    for rep in range(1, R.IVASA_REPEATS + 1):
        cases.append(Case(
            f"IVASA-FAT-r{rep}", "IVASA", R.IVASA_SRC, "ICI-TP §4.1.2", f"IVASA 疲劳：闭眼 5+2 s，30 km/h，第 {rep} 次",
            (lambda ph=0.9 * rep: timeline([Seg(LEAD, speed=30.0), Seg(R.IVASA_FATIGUE_HOLD_S + R.IVASA_EXTRA_HOLD_S, L.UNKNOWN, eyelid=1.0),
                                            Seg(15.0)], 30.0, phase_s=ph)),
            LEAD, lambda c, out: ((PASS if first_onset(out, "acoustic", c.t_ref - 0.5) - c.t_ref <= R.IVASA_FATIGUE_HOLD_S else FAIL),
                                  {"criterion": "告警 ≤ 5 s", "measured_s": round(first_onset(out, "acoustic", c.t_ref - 0.5) - c.t_ref, 3)}),
            {"speed_kph": 30.0, "gaze_label": "unknown (eyes closed)", "location": "-", "movement": "-", "addw_area": "-",
             "required": "Y"}))
    cases += system_cases(cfg)
    return cases


def system_cases(cfg: DmsConfig = PROPOSED) -> list[Case]:
    """Function-level cases for suppression, reset, arbitration, calibration and failure handling."""
    def chk_journey(c, outs):
        j1 = _t_alert(outs[0], c.t_ref)
        j2 = _t_alert(outs[1], c.t_ref) - c.t_ref
        ok = math.isnan(j1) and j2 <= cfg.declared_lga_s + 0.2
        return (PASS if ok else FAIL), {"criterion": "行程1关闭后无报警；行程2自动恢复", "measured_s": round(j2, 3)}

    def chk_adas(c, out):
        t = _t_alert(out, c.t_ref) - c.t_ref
        td = out["t_dec"].to_numpy() - c.t_ref
        fcw = (td > 3.0) & (td <= 4.5)
        quiet = not bool(((out["acoustic"].to_numpy(dtype=bool) | out["haptic"].to_numpy(dtype=bool)) & fcw).any())
        visual = bool((out["visual"].to_numpy(dtype=bool) & fcw).any())
        ok = abs(t - 4.5) <= 0.1 and quiet and visual
        return (PASS if ok else FAIL), {"criterion": "FCW 3.0–4.5 s 期间无 DMS 声/触，只保留视觉；结束后恢复",
                                        "measured_s": round(t, 3)}

    def chk_arb(c, out):
        s = out["source"].astype(str)
        src = s.to_numpy()
        ac = out["acoustic"].to_numpy(dtype=bool)
        t = out["t_dec"].to_numpy()
        dist = ac & s.str.startswith("distraction").to_numpy() & (t >= c.t_ref)
        if not dist.any():
            return FAIL, {"criterion": "分心报警期间不并发疲劳报警，之后补发", "measured_s": float("nan")}
        d0, d1 = t[dist][0], t[dist][-1]
        due = c.meta["drowsy_repeat_due_s"]
        drowsy_during = bool((ac & (src == "drowsy") & (t >= d0) & (t <= d1)).any())
        after = first_onset(out, "acoustic", d1, "drowsy")
        ok = d0 <= due <= d1 and not drowsy_during and not math.isnan(after) and after - d1 <= 0.1
        return (PASS if ok else FAIL), {"criterion": "疲劳重复报警到期时正在分心报警：延后，结束后补发",
                                        "measured_s": round(after - c.t_ref, 3),
                                        "note": f"分心报警 {d0 - c.t_ref:.2f}–{d1 - c.t_ref:.2f} s；疲劳报警到期 {due - c.t_ref:.2f} s"}

    def chk_calib(c, outs):
        early = _t_alert(outs[0], 8.0)
        late = _t_alert(outs[1], 25.0)
        ok = math.isnan(early) and not math.isnan(late)
        return (PASS if ok else FAIL), {"criterion": "标定 20 s 内不报警，之后正常", "measured_s": round(late - 25.0, 3)}

    def chk_fail(c, out):
        t_f = first_onset(out, "failure", c.t_ref - 0.5) - c.t_ref
        f = out["failure"].to_numpy()
        t = out["t"].to_numpy()
        held = bool(f[(t > c.t_ref + 1.5) & (t < c.t_ref + 9.9)].all())
        cleared = not bool(f[t > c.t_ref + 13.0].any())
        no_alert = math.isnan(_t_alert(out, c.t_ref))
        ok = t_f <= cfg.t_fail_confirm_s + LATENCY + 0.05 and held and cleared and no_alert
        return (PASS if ok else FAIL), {"criterion": "无光照 → 恒定故障指示，故障消失后清除", "measured_s": round(t_f, 3)}

    def drowsy(n_cycles: int) -> list[Seg]:          # PERCLOS ~ 0.29: eyes closed 0.5 s every 1.7 s
        return [s for _ in range(n_cycles) for s in (Seg(1.2, L.ROAD, blinks=False), Seg(0.5, L.UNKNOWN, eyelid=1.0))]
    t_glance = 1.0 + 102 * 1.7                        # phone glance starts at 174.4 s
    arb_segs = [Seg(1.0, L.ROAD, speed=80.0, blinks=False)] + drowsy(102) + [Seg(6.0, L.PHONE)] + drowsy(18)
    # first drowsiness alert when the 60 s PERCLOS window is full (t_dec = 60.0 s); repeat due 120 s later
    due = 60.0 + cfg.l2_repeat_s
    return [
        Case("SYS-JOURNEY-RESET", "SYS", R.ADDW_SRC, "Annex I Part 1 §3.1.2, §3.1.6; DE §1.1",
             "行程1：驾驶员关闭 ADDW 报警；行程2：上电后自动恢复",
             lambda: [timeline([Seg(LEAD, speed=55.0, addw_off=True), Seg(5.0, L.PHONE, addw_off=True), Seg(10.0, addw_off=True)], 55.0),
                      timeline([Seg(LEAD, speed=55.0), Seg(5.0, L.PHONE), Seg(10.0)], 55.0)], LEAD, chk_journey,
             {"speed_kph": 55.0, "gaze_label": L.PHONE, "location": "-", "movement": "-", "addw_area": "-", "required": "Y"}),
        Case("SYS-ADAS-SUPPRESS", "SYS", R.ADDW_SRC, "Annex I Part 1 §3.1.5",
             "6 s 手机注视，期间 FCW 在 3.0–4.5 s 报警",
             lambda: timeline([Seg(LEAD, speed=60.0), Seg(3.0, L.PHONE), Seg(1.5, L.PHONE, adas=True), Seg(1.5, L.PHONE), Seg(10.0)], 60.0),
             LEAD, chk_adas,
             {"speed_kph": 60.0, "gaze_label": L.PHONE, "location": "-", "movement": "-", "addw_area": "-", "required": "Y"}),
        Case("SYS-ARBITRATION", "SYS", "EU 2019/2144 GSR", "Art. 6(3)",
             "疲劳（PERCLOS≈0.29）的重复报警到期时刻落在 6 s 手机注视的分心报警期间",
             lambda: timeline(arb_segs, 80.0), t_glance, chk_arb,
             {"speed_kph": 80.0, "gaze_label": L.PHONE, "location": "-", "movement": "-", "addw_area": "-", "required": "Y",
              "drowsy_repeat_due_s": due}),
        Case("SYS-CALIB", "SYS", R.ADDW_SRC, "Annex I Part 1 §3.1.1; DE §1.1",
             "行程开始 8 s 时 5 s 手机注视（标定中）vs 25 s 时",
             lambda: [timeline([Seg(8.0, speed=50.0), Seg(5.0, L.PHONE), Seg(10.0)], 50.0),
                      timeline([Seg(25.0, speed=50.0), Seg(5.0, L.PHONE), Seg(10.0)], 50.0)], 8.0, chk_calib,
             {"speed_kph": 50.0, "gaze_label": L.PHONE, "location": "-", "movement": "-", "addw_area": "-", "required": "Y"}),
        Case("SYS-FAILURE-NOLIGHT", "SYS", R.ADDW_SRC, "Annex I Part 1 §3.5.1.1, §3.5.1.3, §3.5.1.4",
             "传感器无光照 10 s（期间 5 s 手机注视），之后恢复",
             lambda: timeline([Seg(LEAD, speed=60.0), Seg(2.0, camera_ok=False), Seg(5.0, L.PHONE, camera_ok=False),
                               Seg(3.0, camera_ok=False), Seg(15.0)], 60.0), LEAD, chk_fail,
             {"speed_kph": 60.0, "gaze_label": L.PHONE, "location": "-", "movement": "-", "addw_area": "-", "required": "Y"}),
    ]


# ============================================================================ boundary cases
@dataclass
class Boundary:
    test_id: str
    parameter: str
    threshold: str
    value: str
    expect: bool
    build: Callable[[], pd.DataFrame]
    measure: Callable[[pd.DataFrame], float]      # returns event time (nan = no event)
    req: str
    config: DmsConfig = PROPOSED
    note: str = ""


def _alert_after(t0: float, src: str = "distraction", col: str = "acoustic"):
    return lambda out: first_onset(out, col, t0 - 0.5, src)


def boundary_cases() -> list[Boundary]:
    from .config import ADDW_MIN
    B: list[Boundary] = []
    T = LEAD
    # speed gate 20 km/h (ADDW 3.3.2.2 / Euro NCAP DE 1.1)
    for v, exp in ((19.9, False), (20.0, True), (20.1, True)):
        B.append(Boundary(f"BC-SPD-{v:g}", "报警车速门限", "≥ 20 km/h", f"{v:g} km/h", exp,
                          _glance(v, L.PHONE, 6.0), _alert_after(T), "SYS-01"))
    # ADDW_MIN speed band switch at 50 km/h, 4.0 s glance (6.0 s vs 3.5 s threshold)
    for v, exp in ((49.9, False), (50.0, True)):
        B.append(Boundary(f"BC-ADDWMIN-SPD-{v:g}", "ADDW 最低配置速度档", "≥ 50 km/h 用 3.5 s，否则 6 s",
                          f"{v:g} km/h，注视 4.0 s", exp, _glance(v, L.PHONE, 4.0), _alert_after(T), "DIS-02", ADDW_MIN))
    # timer counted regardless of speed (ADDW 3.3.2.3): glance starts at 15 km/h, speed ramps to 30 km/h
    B.append(Boundary("BC-SPD-CROSS", "计时与车速无关", "15→30 km/h 期间注视 6 s", "车速在 2.5 s 时越过 20 km/h", True,
                      lambda: timeline([Seg(LEAD, speed=15.0), Seg(6.0, L.PHONE, speed=15.0, speed_end=51.0), Seg(10.0, speed=30.0)], 15.0),
                      _alert_after(T), "DIS-03",
                      note="越过 20 km/h 时注视已 2.5 s；3.4 s（3.3 s + 时延）时报警，而非从越线时刻重新计时"))
    # long-glance threshold 3.3 s and visual pre-alert 2.0 s
    for d, exp in ((3.2, False), (3.3, True), (3.4, True)):
        B.append(Boundary(f"BC-LD-{d:g}", "长时注视阈值", "≥ 3.3 s", f"注视 {d:g} s", exp, _glance(55.0, L.PHONE, d),
                          _alert_after(T), "DIS-02"))
    for d, exp in ((1.9, False), (2.1, True)):
        B.append(Boundary(f"BC-PRE-{d:g}", "视觉预警阈值", "≥ 2.0 s", f"注视 {d:g} s", exp, _glance(55.0, L.PHONE, d),
                          _alert_after(T, "pre_alert", "visual"), "DIS-09"))
    B.append(Boundary("BC-ADDW-3.5", "ADDW §3.3.2.1 上限", "≤ 3.5 s（试验 ≤ 4.0 s）", "注视 3.5 s @55 km/h", True,
                      _glance(55.0, L.PHONE, 3.5), _alert_after(T), "DIS-02"))
    # interruption tolerance 0.3 s
    for g, exp in ((0.3, True), (1 / 3, False)):
        B.append(Boundary(f"BC-GAP-{g:.2f}", "中断容差", "≤ 0.3 s 不复位", f"2.4 s 手机 + {g:.2f} s 路面 + 2.4 s 手机", exp,
                          (lambda g=g: timeline([Seg(LEAD, speed=55.0), Seg(2.4, L.PHONE), Seg(g, blinks=False), Seg(2.4, L.PHONE),
                                                 Seg(10.0)], 55.0)), _alert_after(T), "DIS-04"))
    B.append(Boundary("BC-BLINK", "眨眼不复位", "unknown ≤ 0.3 s 计入", "手机注视中 0.2 s 眨眼，总 3.6 s", True,
                      lambda: timeline([Seg(LEAD, speed=55.0), Seg(1.7, L.PHONE), Seg(0.2, L.UNKNOWN, eyelid=1.0), Seg(1.7, L.PHONE),
                                        Seg(10.0)], 55.0), _alert_after(T), "DIS-04"))
    B.append(Boundary("BC-UNKNOWN-LONG", "长 unknown 冻结", "> 0.3 s 冻结不复位", "1.8 s 手机 + 1.0 s 不可测 + 2.0 s 手机", True,
                      lambda: timeline([Seg(LEAD, speed=55.0), Seg(1.8, L.PHONE), Seg(1.0, L.UNKNOWN), Seg(2.0, L.PHONE),
                                        Seg(10.0)], 55.0), _alert_after(T), "DIS-04",
                      note="冻结期间不计时：报警比连续注视晚约 1.0 s"))
    # VATS 10 s / 30 s and reset 2.0 s
    for cum, exp in ((9.9, False), (10.1, True)):
        n_full = int(cum // 1.8)
        rest = round(cum - 1.8 * n_full, 3)
        def b(n_full=n_full, rest=rest):
            segs = [Seg(LEAD, speed=70.0)]
            for _ in range(n_full):
                segs += [Seg(1.8, L.INFOTAINMENT), Seg(0.9, blinks=False)]
            segs += [Seg(rest, L.INFOTAINMENT), Seg(15.0)]
            return timeline(segs, 70.0)
        B.append(Boundary(f"BC-VATS-{cum:g}", "VATS 累计阈值", "≥ 10 s / 30 s", f"累计 {cum:g} s", exp, b, _alert_after(T),
                          "DIS-05"))
    for g, exp in ((1.9, True), (2.1, False)):     # 6 x 1.8 s off-road; the 3rd on-road gap is g instead of 0.9 s
        def b_reset(g=g):
            segs = [Seg(LEAD, speed=70.0)]
            for i in range(6):
                segs += [Seg(1.8, L.INFOTAINMENT), Seg(g if i == 2 else 0.9, blinks=False)]
            return timeline(segs + [Seg(15.0)], 70.0)
        B.append(Boundary(f"BC-VATSRESET-{g:g}", "VATS 复位", "路面连续 ≥ 2.0 s 复位",
                          f"6 × 1.8 s 离开（累计 10.8 s），第 3 次路面 {g:g} s", exp, b_reset, _alert_after(T), "DIS-05"))
    for off, on, exp in ((0.6, 1.4, False), (0.8, 1.2, True)):
        B.append(Boundary(f"BC-VATSWIN-{off:g}-{on:g}", "VATS 30 s 窗", "窗内累计 ≥ 10 s",
                          f"{off:g} s 离开 / {on:g} s 路面，持续 60 s（窗内 {30 * off / (off + on):.0f} s）", exp,
                          _pattern(70.0, [L.INFOTAINMENT], off, on, int(60 / (off + on))), _alert_after(T), "DIS-05"))
    # eye closure thresholds
    for d, exp in ((1.1, False), (1.3, True)):
        B.append(Boundary(f"BC-MS-{d:g}", "微睡眠阈值", "闭眼 ≥ 1.2 s", f"闭眼 {d:g} s", exp, _closed(60.0, d),
                          _alert_after(T, "microsleep"), "DRW-05"))
    for d, exp in ((2.9, False), (3.1, True)):
        B.append(Boundary(f"BC-SLEEP-{d:g}", "睡眠阈值", "闭眼 ≥ 3.0 s", f"闭眼 {d:g} s", exp, _closed(60.0, d),
                          _alert_after(T, "sleep"), "DRW-06"))
    for d, exp in ((5.9, False), (6.1, True)):
        B.append(Boundary(f"BC-UNRESP-{d:g}", "无响应阈值", "闭眼 ≥ 6.0 s", f"闭眼 {d:g} s", exp, _closed(60.0, d),
                          _alert_after(T, "", "ef_request"), "DRW-07"))
    # drowsiness speed gate 50 km/h and PERCLOS level 2 at 0.20 (exact closure patterns)
    def perclos_scn(v, closed_f, period_f):
        segs = [Seg(0.5, L.ROAD, speed=v, blinks=False)]
        for _ in range(int(80 * 30 / period_f)):
            segs += [Seg((period_f - closed_f) / 30, L.ROAD, blinks=False), Seg(closed_f / 30, L.UNKNOWN, eyelid=1.0)]
        return lambda: timeline(segs, v)
    for v, exp in ((49.9, False), (50.0, True)):
        B.append(Boundary(f"BC-DROWSY-SPD-{v:g}", "疲劳功能车速门限", "≥ 50 km/h", f"{v:g} km/h，PERCLOS≈0.29", exp,
                          perclos_scn(v, 15, 51), _alert_after(0.0, "drowsy"), "DRW-03"))
    for cf, pf, exp in ((12, 63, False), (12, 57, True)):
        B.append(Boundary(f"BC-PERCLOS-{cf / pf:.3f}", "PERCLOS 报警级", "≥ 0.20", f"PERCLOS {cf / pf:.3f}", exp,
                          perclos_scn(80.0, cf, pf), _alert_after(0.0, "drowsy"), "DRW-02"))
    # availability: inform after 5 s of occlusion
    for d, exp in ((4.9, False), (5.1, True)):
        B.append(Boundary(f"BC-INFORM-{d:g}", "不可用提示", "遮挡 ≥ 5 s 提示（规程 ≤ 10 s）", f"人脸丢失 {d:g} s", exp,
                          (lambda d=d: timeline([Seg(LEAD, speed=50.0), Seg(d, L.UNKNOWN, face=False), Seg(10.0)], 50.0)),
                          _alert_after(T, "", "info"), "AVL-03"))
    # calibration and ADAS coupling
    for d, exp in ((0.7, False), (0.9, True)):
        B.append(Boundary(f"BC-FSS-{d:g}", "ADAS 灵敏度联动", "连续离开路面 ≥ 0.8 s", f"后视镜扫视 {d:g} s", exp,
                          _glance(60.0, L.MIRROR, d), _alert_after(T, "", "fss"), "DIS-08"))
    # end-to-end latency budget (ADDW 4.0 s false-negative limit, Euro NCAP declared 3.4 +/- 0.2 s)
    for lat in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8):
        B.append(Boundary(f"BC-LAT-{lat:g}", "端到端时延预算", "ADDW ≤ 4.0 s；NCAP 3.4 ± 0.2 s", f"时延 {lat:g} s", True,
                          _glance(55.0, L.PHONE, 6.5), _alert_after(T), "DIS-02", note=f"latency={lat}"))
    return B


def evaluate_boundary(b: Boundary) -> dict:
    lat = LATENCY
    if b.note.startswith("latency="):
        lat = float(b.note.split("=")[1])
    out = run(b.build(), b.config, lat)
    t = b.measure(out)
    t0 = LEAD if not b.test_id.startswith(("BC-DROWSY", "BC-PERCLOS")) else 0.0
    occurred = not math.isnan(t)
    rel = round(t - t0, 3) if occurred else float("nan")
    verdict = PASS if occurred == b.expect else FAIL
    extra = ""
    expected_txt, observed_txt = ("是" if b.expect else "否"), ("是" if occurred else "否")
    if b.test_id.startswith("BC-LAT"):
        # expected budget limits: ADDW 3.3 + lat <= 4.0 s -> lat <= 0.7 s; NCAP |3.3 + lat - 3.4| <= 0.2 -> lat <= 0.3 s
        exp_addw = PROPOSED.t_long_s + lat <= R.ADDW_TEST_FN_HIGH_S + 1e-9
        exp_ncap = abs(PROPOSED.t_long_s + lat - PROPOSED.declared_lga_s) <= R.ENCAP_LD_TOL_S + 1e-9
        addw_ok = occurred and rel <= R.ADDW_TEST_FN_HIGH_S + 1e-9
        ncap_ok = occurred and abs(rel - PROPOSED.declared_lga_s) <= R.ENCAP_LD_TOL_S + 1e-9
        yn = lambda x: "是" if x else "否"
        expected_txt = f"ADDW 满足={yn(exp_addw)}；NCAP 满足={yn(exp_ncap)}"
        observed_txt = f"ADDW 满足={yn(addw_ok)}；NCAP 满足={yn(ncap_ok)}"
        verdict = PASS if (addw_ok == exp_addw and ncap_ok == exp_ncap) else FAIL
        extra = "时延预算扫描：判定的是预算边界是否落在计算值上，不是功能通过与否"
    return {"test_id": b.test_id, "requirement": b.req, "config": b.config.name, "parameter": b.parameter,
            "threshold": b.threshold, "value": b.value, "expected_event": expected_txt,
            "observed_event": observed_txt, "event_time_s": rel,
            "note": (b.note if not b.note.startswith("latency=") else "") + extra, "verdict": verdict,
            "data_origin": DATA_ORIGIN}
