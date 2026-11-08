"""Check that every number quoted in README.md is the one in the committed result files.

    python -I scripts/check_readme_numbers.py      # exit code 1 and a list of mismatches if README and results drift

Each expected README snippet is rebuilt from results/*.csv, results/*.json, traceability.csv,
regulation_requirements.csv and the test files, then searched for verbatim in README.md.
"""
from __future__ import annotations

import ast
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

RES = ROOT / "results"
PASS, FAIL, NA = "通过", "不通过", "不适用"


def half_up(x: float, nd: int = 1) -> str:
    return str(Decimal(str(x)).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def pct(x: float) -> str:
    if x == 1.0:
        return "100 %"
    if x == 0.0:
        return "0 %"
    return half_up(float(Decimal(str(x)) * 100), 1) + " %"


def one(values, what: str) -> float:
    u = sorted(set(float(v) for v in values))
    if len(u) != 1:
        raise AssertionError(f"{what}: expected one value, got {u}")
    return u[0]


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / name, encoding="utf-8-sig")


def n_unit_tests() -> int:
    n = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        n += sum(isinstance(x, ast.FunctionDef) and x.name.startswith("test_") for x in tree.body)
    return n


def expected_snippets() -> list[str]:
    out: list[str] = []
    reg = read("results/regulation_tests.csv")
    p = reg[reg.config == "PROPOSED"]
    by_suite = p.suite.value_counts()
    vc = p.verdict.value_counts()
    if vc.get(FAIL, 0):
        raise AssertionError("PROPOSED has failing regulation cases; README says none fail")
    out.append(f"法规用例 {len(p)} 个（ADDW {by_suite['ADDW']}、Euro NCAP {by_suite['ENCAP']}、IVASA {by_suite['IVASA']}、"
               f"功能 {by_suite['SYS']}） | {vc[PASS]} 通过；{vc[NA]} 不适用")
    out.append(f"of {len(p)} protocol cases for the proposed design, {vc[PASS]} pass and {vc[NA]} are not applicable; none fail")
    # three-configuration table
    for cfg in ("PROPOSED", "ADDW_MIN", "NAIVE"):
        cells = []
        for suite in ("ADDW", "ENCAP", "IVASA", "SYS"):
            v = reg[(reg.config == cfg) & (reg.suite == suite)].verdict.value_counts()
            cells.append(f"{v.get(PASS, 0)} / {v.get(FAIL, 0)} / {v.get(NA, 0)}")
        out.append(f"@@ROW {cfg} " + " | ".join(cells))
    # results-at-a-glance rows
    addw_y = p[(p.suite == "ADDW") & (p.required == "Y")]
    out.append(f"声音报警都在 T_away 后 {one(addw_y.t_acoustic_s, 'ADDW acoustic'):g} s")
    ld = p[p.test_id.str.startswith("ENCAP-LD-nondriving")]
    out.append(f"Euro NCAP 长时分心（{len(ld)} 个非驾驶位置） | {one(ld.measured_s, 'ENCAP LD'):g} s")
    vats = p[p.test_id.str.startswith("ENCAP-VATS") & (p.measured_s > 5)]
    phone = p[p.test_id.str.startswith("ENCAP-PHONE")]
    out.append(f"Euro NCAP VATS / 手机使用（{len(vats) + len(phone)} 例） | "
               f"{one(vats.measured_s, 'VATS'):g} / {one(phone.measured_s, 'PHONE'):g} s")
    m = p.set_index("test_id")
    windows = [m.loc[t, "criterion"].split()[-2] for t in ("ENCAP-UNRESP-sleep", "ENCAP-UNRESP-lap")]
    out.append(f"{m.loc['ENCAP-MS', 'measured_s']:g} / {m.loc['ENCAP-SLEEP', 'measured_s']:g} / "
               f"{m.loc['ENCAP-UNRESP-sleep', 'measured_s']:g} / {m.loc['ENCAP-UNRESP-lap', 'measured_s']:g} s"
               f"（判定窗 1.3 ± 0.2、3 ± 0.2、{windows[0]}、{windows[1]} s）")
    occ = [m.loc[t, "measured_s"] for t in ("ENCAP-NF-hand", "ENCAP-OCC-sunglasses-inform")]
    out.append(f"{min(occ):.1f}–{max(occ):.1f} s 提示“DMS 受限”")
    bnd = read("results/boundary_tests.csv")
    if not (bnd.verdict == PASS).all():
        raise AssertionError("boundary cases are not all passing; README says all as designed")
    out.append(f"边界用例 {len(bnd)} 个 | 全部按设计")
    # false-alarm study
    fa = read("results/false_alarm_summary.csv").set_index("config")
    P = fa.loc["PROPOSED"]
    n_ev = int(P.long_glance_n + P.phone_vats_n)
    assert P.long_glance_on_time_encap == 1.0 and P.phone_vats_on_time_encap == 1.0
    out.append(f"误报研究：{P.hours:g} h 合成自然驾驶，{n_ev} 个注入分心事件 | 误报 {P.false_alarms_per_h:g} 次/h；"
               f"{n_ev} 个事件全部按 Euro NCAP 时限检出；长注视中位时延 {P.long_glance_latency_median_s:.2f} s"
               f"（P95 {P.long_glance_latency_p95_s:.2f} s）")
    out.append(f"误报研究（合成自然驾驶，{P.hours:g} h / 配置，每种配置 {n_ev} 个注入分心事件）")
    out.append(f"全部 {len(fa)} 种配置见")
    G = fa.loc["gap=0.6s"]
    out.append(f"synthetic {P.hours:g} h false-alarm study the proposed design gives {P.false_alarms_per_h:g} false alarms "
               f"per hour, while widening the glance-interruption tolerance from 0.3 s to 0.6 s gives {G.false_alarms_per_h:g} per hour")
    labels = {"PROPOSED": "PROPOSED", "ADDW_MIN": "ADDW_MIN", "T_long=3.0": "T_long = 3.0 s", "T_long=3.5": "T_long = 3.5 s",
              "reset=1.0s": "复位 = 1.0 s", "gap=0.05s": "容差 = 0.05 s", "gap=0.6s": "容差 = 0.6 s",
              "pre=2.5s": "视觉预警 = 2.5 s"}
    for cfg, lab in labels.items():
        r = fa.loc[cfg]
        vis = "0" if r.visual_pre_alerts_per_h == 0 else half_up(r.visual_pre_alerts_per_h)
        lat = "—" if pd.isna(r.phone_vats_latency_median_s) else half_up(r.phone_vats_latency_median_s) + " s"
        out.append(f"| {lab} | {r.false_alarms_per_h:g} | {r.justified_natural_per_h:.2f} | {vis} | "
                   f"{pct(r.long_glance_on_time_addw)} / {pct(r.long_glance_on_time_encap)} | "
                   f"{pct(r.phone_vats_detect_rate)} / {pct(r.phone_vats_on_time_encap)} | {lat} |")
    ev = read("results/false_alarm_events.csv")
    g = ev[(ev.config == "gap=0.6s") & (ev.classification == "false-alarm")]
    assert len(g) == G.false_alarms
    out.append(f"{len(g)} 次误报里有 {(g.gt_activity_at_alert == 'driving').sum()} 次来自路口扫视和回头看")
    out.append(f"视觉预警每小时 {int(P.visual_pre_alerts_per_h)} 次")
    out.append(f"改成 2.5 s 可降到 {int(fa.loc['pre=2.5s'].visual_pre_alerts_per_h)} 次")
    # NHTSA
    nh = read("results/nhtsa_summary.csv").set_index("variant")
    npart = read("results/nhtsa_participants.csv").groupby("variant").size()
    names = {"touch_address": "触屏输入地址", "voice_address": "语音输入地址", "touch_favourite": "触屏选收藏地址"}
    for v, lab in names.items():
        r, n = nh.loc[v], int(npart[v])
        verdict = "可接受" if r.task_acceptable else "不可接受 → 行驶中锁止"
        out.append(f"| {lab} | {r.n_meet_a}/{n} | {r.n_meet_b}/{n} | {r.n_meet_c}/{n} | {verdict} | "
                   f"{half_up(r.median_teort_s)} / {half_up(r.max_teort_s)} s | {round(r.share_with_dms_alert * n)}/{n} |")
    t, vo = nh.loc["touch_address"], nh.loc["voice_address"]
    out.append(f"触屏：总离路时间准则只有 {t.n_meet_c}/{int(npart['touch_address'])} 人满足")
    out.append(f"语音：三条准则 {vo.n_meet_a}/{vo.n_meet_b}/{vo.n_meet_c} 人满足")
    out.append(f"NHTSA 视觉-手动指南的条款整理成需求（共 {read('traceability.csv').req_id.nunique()} 条）")
    out.append(f"clauses into {read('traceability.csv').req_id.nunique()} requirements")
    # traceability and tests
    tr = read("traceability.csv")
    rq = read("regulation_requirements.csv")
    ut = tr[tr.test_id.str.startswith("UT:")].test_id.nunique()
    n_ut = n_unit_tests()
    if ut != n_ut:
        raise AssertionError(f"traceability.csv has {ut} unit tests, tests/ defines {n_ut}: rerun scripts/run_all.py")
    if (tr[tr.test_id.str.startswith("UT:")].verdict != PASS).any():
        raise AssertionError("traceability.csv has unit-test rows that are not 通过 (run pytest with --junitxml first)")
    scen = tr[~tr.test_id.str.startswith(("UT:", "N/A:"))].test_id.nunique()
    out.append(f"{len(rq)} 条条款 → {tr.req_id.nunique()} 条需求 → {scen} 个场景/研究结果 + {n_ut} 个单元测试，{len(tr)} 行")
    out.append(f"traceability.csv             条款 → 需求 → 测试 → 结果（{len(tr)} 行，双向完整）")
    out.append(f"regulation_requirements.csv  {len(rq)} 条条款")
    vv = rq.verified.value_counts()
    out.append(f"{vv.get('Y', 0)} 条已核实、{vv.get('N', 0)} 条未核实")
    out.append(f"`pytest` {n_ut} 项全部通过")
    out.append(f"再跑 {n_ut} 项 pytest")
    out.append(f"tests/                       {n_ut} 项 pytest")
    return out


def main() -> int:
    readme = (ROOT / "README.md").read_text(encoding="utf-8").replace("**", "")
    lines = readme.splitlines()
    missing = []
    for s in expected_snippets():
        if s.startswith("@@ROW "):
            cfg, cells = s[6:].split(" ", 1)
            row = [ln for ln in lines if ln.startswith(f"| {cfg} |")]
            if not any(r.rstrip().endswith(" | " + cells + " |") for r in row):
                missing.append(f"config table row {cfg}: expected '... | {cells} |'")
        elif s not in readme:
            missing.append(s)
    if missing:
        print(f"README.md: {len(missing)} number(s) do not match the result files:")
        for s in missing:
            print("  - expected:", s)
        return 1
    print("README numbers OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
