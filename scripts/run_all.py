"""Run every study and write results/ (CSV, JSON, tables.md, figures) and traceability.csv.

    python -I scripts/run_all.py            # about 2-3 minutes, single process
    python -I -m pytest -q tests --junitxml=results/pytest_junit.xml
    python -I scripts/make_traceability.py  # refresh traceability.csv with the unit-test verdicts

All inputs are SYNTHETIC (dms/scenarios.py); every results row carries data_origin = SYNTHETIC.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from dms import ddaw, false_alarm, nhtsa, plots, trace  # noqa: E402
from dms.config import ADDW_MIN, NAIVE, PROPOSED, parameter_rows  # noqa: E402
from dms.scenarios import DATA_ORIGIN, Seg, timeline  # noqa: E402
from dms.suites import boundary_cases, evaluate, evaluate_boundary, regulation_cases  # noqa: E402
from dms.system import run  # noqa: E402
from dms import labels as L  # noqa: E402

RES = ROOT / "results"


def save(df: pd.DataFrame, name: str) -> None:
    df.to_csv(RES / name, index=False, encoding="utf-8-sig")


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join("-" if (isinstance(v, float) and v != v) else (f"{v:g}" if isinstance(v, float) else str(v))
                                     for v in r) + " |")
    return "\n".join(out)


def main() -> int:
    RES.mkdir(exist_ok=True)
    t_start = time.perf_counter()
    # ------------------------------------------------------------------ 1 regulation cases, three presets
    reg = pd.DataFrame([evaluate(c, cfg) for cfg in (PROPOSED, ADDW_MIN, NAIVE) for c in regulation_cases(cfg)])
    save(reg, "regulation_tests.csv")
    print(f"regulation cases: {len(reg)} rows ({time.perf_counter() - t_start:.0f} s)")
    # ------------------------------------------------------------------ 2 boundary cases
    bnd = pd.DataFrame([evaluate_boundary(b) for b in boundary_cases()])
    save(bnd, "boundary_tests.csv")
    print(f"boundary cases: {len(bnd)} ({time.perf_counter() - t_start:.0f} s)")
    # ------------------------------------------------------------------ 3 false-alarm study
    ev, ep, fa = false_alarm.study()
    save(ev, "false_alarm_events.csv")
    save(ep, "false_alarm_episodes.csv")
    save(fa, "false_alarm_summary.csv")
    print(f"false-alarm study: {len(fa)} configurations ({time.perf_counter() - t_start:.0f} s)")
    # ------------------------------------------------------------------ 4 NHTSA HMI evaluation
    nh_per, nh_sum = nhtsa.evaluate()
    save(nh_per, "nhtsa_participants.csv")
    save(nh_sum, "nhtsa_summary.csv")
    # ------------------------------------------------------------------ 5 DDAW Part 2 computation demo
    dd_per, dd_sum = ddaw.synthetic_validation()
    save(dd_per, "ddaw_synthetic_participants.csv")
    (RES / "ddaw_synthetic_summary.json").write_text(json.dumps(dd_sum, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"NHTSA + DDAW demo done ({time.perf_counter() - t_start:.0f} s)")
    # ------------------------------------------------------------------ 6 parameters with provenance
    save(pd.DataFrame(parameter_rows()), "parameters.csv")
    # ------------------------------------------------------------------ 7 figures
    segs = [Seg(60.0, speed=55.0), Seg(6.0, L.PHONE), Seg(12.0)]
    for _ in range(7):
        segs += [Seg(1.8, L.PHONE), Seg(0.9, blinks=False)]
    segs.append(Seg(14.0))
    scn = timeline(segs, 55.0)
    out = run(scn, PROPOSED, 0.1)
    plots.fig_timeline(out, scn, RES / "fig1_timeline.png", 56.0, 111.0,
                       [(60.0, "6 s phone glance"), (78.0, "phone VATS 1.8 s / 0.9 s")])
    plots.fig_false_alarm(fa, RES / "fig2_false_alarm_tradeoff.png")
    plots.fig_nhtsa(nh_per, nh_sum, RES / "fig3_nhtsa_hmi.png")
    # ------------------------------------------------------------------ 8 tables for the docs
    p = reg[reg.config == "PROPOSED"]
    t1 = (reg.groupby(["config", "suite"]).verdict.value_counts().unstack(fill_value=0).reset_index())
    t2 = p[p.suite == "ADDW"][["test_id", "location", "speed_kph", "addw_area", "required", "t_visual_s", "t_acoustic_s",
                               "t_haptic_s", "criterion", "verdict"]]
    t3 = p[p.suite != "ADDW"][["test_id", "speed_kph", "criterion", "measured_s", "verdict"]]
    t4 = bnd[["test_id", "value", "expected_event", "observed_event", "event_time_s", "verdict"]]
    t5 = fa[["config", "false_alarms_per_h", "justified_natural_per_h", "visual_pre_alerts_per_h",
             "long_glance_on_time_addw", "long_glance_on_time_encap", "long_glance_latency_median_s",
             "phone_vats_detect_rate", "phone_vats_on_time_encap", "phone_vats_latency_median_s", "fss_time_share"]]
    t6 = nh_sum.drop(columns=["data_origin"])
    tables = ["<!-- generated by scripts/run_all.py; all numbers from SYNTHETIC inputs -->",
              "## T1 法规用例判定计数（三种配置）", md_table(t1),
              "## T2 ADDW Part 2 抽查（PROPOSED，时间从 T_away 起算）", md_table(t2),
              "## T3 Euro NCAP / IVASA / 功能用例（PROPOSED）", md_table(t3),
              "## T4 边界用例", md_table(t4),
              "## T5 误报研究", md_table(t5),
              "## T6 NHTSA 扫视准则", md_table(t6),
              "## T7 DDAW 方法演示", md_table(pd.DataFrame([dd_sum]))]
    (RES / "tables.md").write_text("\n\n".join(tables) + "\n", encoding="utf-8")
    # ------------------------------------------------------------------ 9 summary.json
    prop_fa = fa.set_index("config").loc["PROPOSED"]
    summary = {
        "data_origin": DATA_ORIGIN,
        "regulation_cases_proposed": p.verdict.value_counts().to_dict(),
        "regulation_cases_by_config": {c: g.verdict.value_counts().to_dict() for c, g in reg.groupby("config")},
        "boundary_cases": bnd.verdict.value_counts().to_dict(),
        "false_alarm_proposed": {k: (None if v != v else v) for k, v in prop_fa.to_dict().items()
                                 if isinstance(v, (int, float))},
        "nhtsa": nh_sum.drop(columns=["data_origin"]).to_dict(orient="records"),
        "ddaw_demo": dd_sum,
        "runtime_s": round(time.perf_counter() - t_start, 1),
    }
    (RES / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    # ------------------------------------------------------------------ 10 traceability
    problems = trace.build()
    print(f"traceability: {'OK' if not problems else str(len(problems)) + ' problems'}")
    for pr in problems:
        print("  -", pr)
    print(f"done in {time.perf_counter() - t_start:.0f} s")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
