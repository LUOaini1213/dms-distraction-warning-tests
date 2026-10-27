"""DMS parameters.  Every field is either a regulatory value (clause in PROVENANCE) or a DESIGN ASSUMPTION (DA-xx).

Presets
  PROPOSED  the design of 需求规格.md: one 3.3 s long-glance threshold for all speeds >= 20 km/h, Euro NCAP VATS,
            driving-task glances (mirror / instrument) never warned but couple the ADAS sensitivity.
  ADDW_MIN  the bare ADDW minimum: 3.5 s at >= 50 km/h, 6.0 s at 20-50 km/h, no VATS, 50 ms tolerance.
  NAIVE     a simple eyes-off-road timer: every non-road label (incl. mirror, instrument, unknown) counts and
            any interruption resets - used only to show why the PROPOSED rules exist.
The configuration is frozen: detection sensitivity cannot be changed at run time (Euro NCAP DE 1.1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, fields, replace

from . import labels as L
from . import regulations as R


@dataclass(frozen=True)
class DmsConfig:
    name: str = "PROPOSED"
    dt: float = 1.0 / 30.0                    # DA-01 gaze/eyelid output rate 30 Hz
    # ---- activation / gating
    v_distraction_kph: float = R.ADDW_ACTIVATION_KPH       # ADDW 3.1.1/3.3.2.2, Euro NCAP DE 1.1
    v_drowsy_kph: float = R.ENCAP_IMPAIR_MIN_SPEED_KPH     # Euro NCAP DE 1.3.2 (DDAW 3.1.4 needs only >70)
    calib_time_s: float = 20.0                # DA-22 calibration time (<= ADDW 60 s, <= Euro NCAP 1 min)
    calib_speed_kph: float = R.ENCAP_START_MEASURE_KPH     # Euro NCAP DE 1.1 (>= 10 km/h)
    # ---- long single glance (ADDW 3.3.2, Euro NCAP DE 1.2.1)
    t_long_s: float = 3.3                     # DA-04 = ADDW 3.5 s minus a 0.2 s end-to-end latency budget
    t_long_low_s: float | None = None         # threshold for v_distraction <= v < v_high (None = t_long_s)
    v_high_kph: float = R.ADDW_HIGH_SPEED_KPH
    t_pre_s: float | None = 2.0               # DA-05 visual pre-alert (NHTSA 2.0 s single-glance reference)
    gap_tol_s: float = 0.3                    # DA-03 interruption tolerance (>= ADDW 3.3.2.4 minimum 50 ms)
    # ---- VATS cumulative glance-away (Euro NCAP DE 1.2.2 / 1.2.3)
    vats_enabled: bool = True
    vats_window_s: float = R.ENCAP_VATS_WINDOW_S
    vats_threshold_s: float = R.ENCAP_VATS_CUM_S
    vats_pre_s: float | None = 7.0            # DA-06 visual pre-alert on cumulative time
    vats_reset_s: float = 2.0                 # DA-06 T_VATS_reset: continuous road gaze that ends an episode
    # ---- which labels count
    counted_regions: frozenset = L.NON_DRIVING
    unknown_counts: bool = False              # NAIVE: unknown counted as eyes-off-road
    # ---- escalation (ADDW 3.4.1.1 cascade, 3.4.1.2 adaptation)
    t_haptic_after_s: float = 1.5             # DA-07 haptic if the trigger persists 1.5 s after acoustic onset
    repeat_window_s: float = 60.0             # DA-08 a new warning within 60 s of the last one starts haptic
    min_acoustic_s: float = 1.0               # DA-09 audible part long enough to be understood (ADDW 3.4.3.4)
    visual_hold_s: float = 2.0                # DA-09 visual tell-tale held after the trigger ceases
    t_unresp_gaze_s: float = R.ENCAP_UNRESP_GAZE_S          # Euro NCAP DE 1.3.5
    # ---- drowsiness
    eyelid_closed: float = 0.8                # DA-10 eyelid closure >= 0.8 counts as closed (P80 convention)
    perclos_window_s: float = 60.0            # DA-11
    perclos_l1: float = 0.12                  # DA-12 information level
    perclos_l1_exit: float = 0.08
    perclos_l2: float = 0.20                  # DA-12 warning level, assumed ~ KSS >= 8 (NOT validated)
    perclos_l2_exit: float = 0.15
    l2_alert_s: float = 3.0                   # DA-13 length of one drowsiness acoustic alert
    l2_repeat_s: float = 120.0                # DA-13 repeat while still at level 2
    t_microsleep_s: float = 1.2               # DA-14 within Euro NCAP DE 1.3.3 range 1-2 s
    t_sleep_s: float = R.ENCAP_SLEEP_S        # Euro NCAP DE 1.3.4
    t_unresp_eyes_s: float = R.ENCAP_UNRESP_EYES_S          # Euro NCAP DE 1.3.5
    microsleep_hold_s: float = 2.0            # DA-14 keep the microsleep warning 2 s after the eyes open
    microsleep_to_l2: int = 2                 # DA-15 two microsleeps within 5 min raise level 2
    microsleep_memory_s: float = 300.0        # DA-15
    # ---- availability (ADDW 3.5, Euro NCAP DE 1.1.1.2)
    t_inform_s: float = 5.0                   # DA-16 inform the driver after 5 s (Euro NCAP: within 10 s)
    t_fail_confirm_s: float = 1.0             # DA-16 debounce for camera fault / no light
    t_recover_s: float = 2.0                  # DA-16 valid tracking needed to clear the information
    # ---- ADAS (FCW/AEB) sensitivity coupling (Euro NCAP DE 1.4.3)
    t_fss_on_s: float = 0.8                   # DA-17 continuous gaze away (<= 1.0 s required)
    t_fss_off_road_s: float = R.ENCAP_FSS_OFF_ROAD_S
    t_fss_off_state_s: float = R.ENCAP_FSS_OFF_STATE_S
    # ---- values that would be declared in a Euro NCAP dossier (threshold + nominal 0.1 s latency, DA-02)
    declared_lga_s: float = 3.4
    declared_microsleep_s: float = 1.3
    declared_vats_s: float = R.ENCAP_VATS_CUM_S

    def n(self, seconds: float | None) -> int:
        """Frames needed to reach `seconds` (integer counters avoid float drift)."""
        if seconds is None:
            return 10 ** 9
        return max(1, int(math.ceil(seconds / self.dt - 1e-6)))

    def with_(self, **kw) -> "DmsConfig":
        return replace(self, **kw)


PROPOSED = DmsConfig()
ADDW_MIN = DmsConfig(name="ADDW_MIN", t_long_s=R.ADDW_T_MAX_HIGH_S, t_long_low_s=R.ADDW_T_MAX_LOW_S,
                     t_pre_s=None, gap_tol_s=R.ADDW_SACCADE_TOL_MIN_S, vats_enabled=False, vats_pre_s=None,
                     declared_lga_s=R.ADDW_T_MAX_HIGH_S + 0.1)
NAIVE = DmsConfig(name="NAIVE", counted_regions=L.EYES_OFF_ROAD, unknown_counts=True, gap_tol_s=0.0,
                  t_pre_s=None, vats_pre_s=None)
PRESETS = {c.name: c for c in (PROPOSED, ADDW_MIN, NAIVE)}

# Provenance of every PROPOSED parameter: regulatory clause or design assumption (written to parameters.csv)
PROVENANCE: dict[str, str] = {
    "dt": "DA-01 30 Hz 注视/眼睑输出（Euro NCAP SD-202 §1 只规定试验录像 ≥25 Hz）",
    "v_distraction_kph": "EU 2023/2590 ADDW Annex I Part 1 §3.1.1, §3.3.2.2; Euro NCAP DE v1.2 §1.1",
    "v_drowsy_kph": "Euro NCAP DE v1.2 §1.1, §1.3.2（DDAW §3.1.4 只要求 >70 km/h 激活）",
    "calib_time_s": "DA-22 标定 20 s（ADDW §3.1.1 ≤60 s；Euro NCAP DE §1.1 ≤1 min）",
    "calib_speed_kph": "Euro NCAP DE v1.2 §1.1（≥10 km/h）",
    "t_long_s": "DA-04 = ADDW §3.3.2.1 的 3.5 s 减 0.2 s 端到端时延预算；落在 Euro NCAP §1.2.1 的 3–4 s 内",
    "t_long_low_s": "DA-04 PROPOSED 不分速度档（ADDW_MIN 预设用 ADDW §3.3.2.2 的 6 s）",
    "v_high_kph": "EU 2023/2590 ADDW §3.3.2.1 (a)",
    "t_pre_s": "DA-05 视觉预警 2.0 s（参考 NHTSA VI.E.14 的 2.0 s 单次扫视准则）",
    "gap_tol_s": "DA-03 中断容差 0.3 s（ADDW §3.3.2.4 要求 ≥50 ms）",
    "vats_enabled": "Euro NCAP DE v1.2 §1.2.2（ADDW 未要求，见其序言 (3)）",
    "vats_window_s": "Euro NCAP DE v1.2 §1.2.2",
    "vats_threshold_s": "Euro NCAP DE v1.2 §1.2.2",
    "vats_pre_s": "DA-06 累计 7 s 视觉预警",
    "vats_reset_s": "DA-06 T_VATS_reset = 2.0 s（Euro NCAP SD-202 §4.1.1.2 由厂家申报）",
    "counted_regions": "DA-18 计入分心的标签：infotainment / phone / off_road",
    "unknown_counts": "DA-03 unknown 不计入（超过容差冻结计时）",
    "t_haptic_after_s": "DA-07 声学后 1.5 s 仍未消除 → 触觉（ADDW §3.4.1.1 允许逐级加强）",
    "repeat_window_s": "DA-08 60 s 内再次报警直接触觉（ADDW §3.4.1.2 允许按历史事件调整）",
    "min_acoustic_s": "DA-09 声学最短 1.0 s（ADDW §3.4.3.4 要求“足以理解”）",
    "visual_hold_s": "DA-09 视觉保持 2.0 s",
    "t_unresp_gaze_s": "Euro NCAP DE v1.2 §1.3.5",
    "eyelid_closed": "DA-10 眼睑闭合度 ≥0.8 记为闭眼（P80 约定）",
    "perclos_window_s": "DA-11 PERCLOS 窗口 60 s（DDAW Part 2 §5.2.1(a) 只点名 PERCLOS，无阈值）",
    "perclos_l1": "DA-12 信息级",
    "perclos_l1_exit": "DA-12 信息级退出（滞回）",
    "perclos_l2": "DA-12 报警级，假设对应 KSS≥8（DDAW §3.3.1），未经标定",
    "perclos_l2_exit": "DA-12 报警级退出（滞回）",
    "l2_alert_s": "DA-13 单次疲劳声学报警时长",
    "l2_repeat_s": "DA-13 疲劳报警重复间隔",
    "t_microsleep_s": "DA-14 1.2 s，在 Euro NCAP DE §1.3.3 的 1–2 s 内",
    "t_sleep_s": "Euro NCAP DE v1.2 §1.3.4；SD-202 §1.1 固定 3 s",
    "t_unresp_eyes_s": "Euro NCAP DE v1.2 §1.3.5",
    "microsleep_hold_s": "DA-14 睁眼后保持 2 s",
    "microsleep_to_l2": "DA-15 5 min 内 2 次微睡眠 → 疲劳报警级",
    "microsleep_memory_s": "DA-15",
    "t_inform_s": "DA-16 5 s 提示（Euro NCAP DE §1.1.1.2 要求 10 s 内）",
    "t_fail_confirm_s": "DA-16 故障/无光照去抖 1.0 s（ADDW §3.5.1.3）",
    "t_recover_s": "DA-16 恢复判定 2.0 s",
    "t_fss_on_s": "DA-17 0.8 s（Euro NCAP DE §1.4.3.1 要求 ≤1 s）",
    "t_fss_off_road_s": "Euro NCAP DE v1.2 §1.4.3 表（≥1 s 路面注视后结束）",
    "t_fss_off_state_s": "Euro NCAP DE v1.2 §1.4.3 表（状态结束 ≥2 s 后结束）",
    "declared_lga_s": "DA-02 申报值 = 3.3 s 阈值 + 0.1 s 名义时延；须在 Euro NCAP 3–4 s 内",
    "declared_microsleep_s": "DA-02 申报值 = 1.2 s + 0.1 s；须在 Euro NCAP 1–2 s 内",
    "declared_vats_s": "Euro NCAP DE v1.2 §1.2.2",
    "name": "-",
}


# Design assumptions that describe models and targets rather than DMS parameters
MODEL_ASSUMPTIONS: dict[str, str] = {
    "DA-19": "合成自然驾驶模型：半马尔可夫扫视（道路停留中位 3.5 s）、眨眼 17 次/min、标签闪变/跟踪丢失/整段误分类、"
             "驾驶相关情境（变道、汇入回头看、路口左右扫视、导航查看、设定 ACC），见 dms/scenarios.py NaturalParams",
    "DA-20": "NHTSA 任务模型：各步骤基准扫视时长、年龄系数 1.00/1.00/1.10/1.25、个体系数 σ=0.12、单次散布 σ=0.20、"
             "地址需输入 14 个字符、每次扫视输入 2–4 个字符；语音方案识别错误率 12 %",
    "DA-21": "DDAW 演示用 KSS→眼睑模型：眨眼 0.3 次/s，闭眼比例 0.03 + 0.045·max(0, KSS−5)^1.3，个体敏感度 σ=0.3（虚构）",
    "DA-23": "设计目标：合成自然驾驶中误报 ≤1 次/h，且注入的分心事件 100 % 按 Euro NCAP 时限检出",
}


def parameter_rows(cfg: DmsConfig = PROPOSED) -> list[dict]:
    rows = []
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if isinstance(v, frozenset):
            v = " / ".join(sorted(v))
        prov = PROVENANCE.get(f.name, "?")
        kind = "-" if prov == "-" else ("设计假设" if "DA-" in prov else "法规/规程")
        rows.append({"parameter": f.name, "value": v, "provenance": prov, "kind": kind})
    for tag, text in MODEL_ASSUMPTIONS.items():
        rows.append({"parameter": tag, "value": "-", "provenance": f"{tag} {text}", "kind": "设计假设（模型/目标）"})
    return rows
