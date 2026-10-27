"""Build traceability.csv (clause -> requirement -> test -> result) and check completeness in both directions.

Checks (each failure is reported; build() returns the list of problems):
  clause -> requirement   every row of regulation_requirements.csv is cited by >= 1 requirement
  requirement -> clause   every cited "source|clause" exists in regulation_requirements.csv; every DA tag exists
  requirement -> test     every requirement has >= 1 test (scenario result or unit test) or an N/A reason
  test -> requirement     every scenario result ID and every unit test is linked to >= 1 requirement;
                          every selector matches >= 1 result (no dead links)
  parameter -> source     every PROPOSED parameter has a clause or DA-xx provenance; every DA-xx is used
"""
from __future__ import annotations

import ast
import csv
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

from .config import MODEL_ASSUMPTIONS, PROVENANCE, parameter_rows
from .requirements import REQS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
CSV_CLAUSES = ROOT / "regulation_requirements.csv"
OUT = ROOT / "traceability.csv"


def clause_keys() -> set[str]:
    d = pd.read_csv(CSV_CLAUSES, encoding="utf-8-sig")
    return {f"{s}|{c}" for s, c in zip(d.source, d.clause)}


def da_tags() -> set[str]:
    return {m for v in PROVENANCE.values() for m in re.findall(r"DA-\d+", v)} | set(MODEL_ASSUMPTIONS)


def unit_tests() -> dict[str, list[str]]:
    """Unit test name -> requirement IDs from the 'REQ:' line of its docstring."""
    out = {}
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                doc = ast.get_docstring(node) or ""
                m = re.search(r"REQ:\s*([A-Z0-9 ,\-]+)", doc)
                out[f"UT:{f.name}::{node.name}"] = [x.strip() for x in m.group(1).split(",")] if m else []
    return out


def junit_verdicts() -> dict[str, str]:
    p = RES / "pytest_junit.xml"
    if not p.exists():
        return {}
    res = {}
    for tc in ET.parse(p).getroot().iter("testcase"):
        fname = tc.get("classname", "").split(".")[-1] + ".py"
        verdict = "通过"
        if tc.find("failure") is not None or tc.find("error") is not None:
            verdict = "不通过"
        elif tc.find("skipped") is not None:
            verdict = "跳过"
        res[f"UT:{fname}::{tc.get('name')}"] = verdict
    return res


def _fmt(x) -> str:
    if isinstance(x, float):
        return "-" if x != x else f"{x:g}"
    return str(x)


def scenario_results() -> dict[str, dict]:
    res: dict[str, dict] = {}
    p = RES / "regulation_tests.csv"
    if p.exists():
        d = pd.read_csv(p, encoding="utf-8-sig")
        for r in d[d.config == "PROPOSED"].itertuples():
            res[r.test_id] = {"kind": f"法规用例（{r.suite}）", "result": f"{r.criterion}；实测 {_fmt(r.measured_s)} s",
                              "verdict": r.verdict, "evidence": "results/regulation_tests.csv"}
    p = RES / "boundary_tests.csv"
    if p.exists():
        for r in pd.read_csv(p, encoding="utf-8-sig").itertuples():
            res[r.test_id] = {"kind": "边界用例", "result": f"{r.value}：期望 {r.expected_event} / 实测 {r.observed_event}（{_fmt(r.event_time_s)} s）",
                              "verdict": r.verdict, "evidence": "results/boundary_tests.csv"}
    p = RES / "false_alarm_summary.csv"
    if p.exists():
        for r in pd.read_csv(p, encoding="utf-8-sig").itertuples():
            txt = (f"误报 {r.false_alarms_per_h:g}/h；长注视检出 {r.long_glance_detect_rate:.0%}、按时(NCAP) {r.long_glance_on_time_encap:.0%}；"
                   f"VATS 检出 {r.phone_vats_detect_rate:.0%}、中位时延 {_fmt(r.phone_vats_latency_median_s)} s")
            if r.config == "PROPOSED":
                ok = r.false_alarms_per_h <= 1.0 and r.long_glance_on_time_encap == 1.0 and r.phone_vats_on_time_encap == 1.0
                verdict = "通过" if ok else "不通过"
                txt += "（设计目标 DA-23：误报 ≤1 次/h 且注入分心 100 % 按时检出）"
            else:
                verdict = "参考（对比配置）"
            res[f"FA-{r.config}"] = {"kind": "误报研究（合成）", "result": txt, "verdict": verdict,
                                     "evidence": "results/false_alarm_summary.csv"}
    p = RES / "nhtsa_summary.csv"
    if p.exists():
        for r in pd.read_csv(p, encoding="utf-8-sig").itertuples():
            ok = bool(r.task_acceptable)
            res[f"NH-{r.variant}"] = {
                "kind": "NHTSA 扫视评估（合成）",
                "result": f"满足 a/b/c 的人数 {r.n_meet_a}/{r.n_meet_b}/{r.n_meet_c}（需 ≥21）；TEORT 中位 {r.median_teort_s:g} s",
                "verdict": "可接受" if ok else "不可接受 → 行驶中锁止", "evidence": "results/nhtsa_summary.csv"}
    p = RES / "ddaw_synthetic_summary.json"
    if p.exists():
        s = json.loads(p.read_text(encoding="utf-8"))
        res["DD-SYN"] = {"kind": "DDAW 验证方法演示（合成）",
                         "result": f"计入 {s.get('participants_counted')} 人，TP+FN {s.get('tp_plus_fn')}；平均灵敏度 {s.get('mean_sensitivity')}，下界 {s.get('lower_bound')}",
                         "verdict": "演示（非验证）", "evidence": "results/ddaw_synthetic_summary.json"}
    return res


def _match(sel: str, ids) -> list[str]:
    if sel.endswith("*"):
        return [i for i in ids if i.startswith(sel[:-1])]
    return [i for i in ids if i == sel]


def build(write: bool = True) -> list[str]:
    problems: list[str] = []
    keys = clause_keys()
    das = da_tags()
    results = scenario_results()
    uts = unit_tests()
    junit = junit_verdicts()
    req_ids = {r.rid for r in REQS}
    cited_clauses, cited_das, linked_tests = set(), set(), set()
    rows = []
    ut_by_req: dict[str, list[str]] = {}
    for ut, reqs in uts.items():
        if not reqs:
            problems.append(f"单元测试没有 REQ 标注：{ut}")
        for r in reqs:
            if r not in req_ids:
                problems.append(f"单元测试引用了不存在的需求 {r}：{ut}")
            ut_by_req.setdefault(r, []).append(ut)
    for req in REQS:
        for c in req.clauses:
            if c not in keys:
                problems.append(f"{req.rid} 引用的条款不在 regulation_requirements.csv：{c}")
            cited_clauses.add(c)
        for d in req.da:
            if d not in das:
                problems.append(f"{req.rid} 引用的设计假设不存在：{d}")
            cited_das.add(d)
        src = " ; ".join(req.clauses) if req.clauses else "设计假设"
        base = {"req_id": req.rid, "requirement": f"{req.title}：{req.text}", "kind": req.kind, "source_clauses": src,
                "design_assumptions": ", ".join(req.da) or "-"}
        tests: list[str] = []
        for sel in req.tests:
            hit = _match(sel, results)
            if not hit:
                problems.append(f"{req.rid} 的测试选择器没有匹配到结果：{sel}")
            tests += [h for h in hit if h not in tests]
        for t in tests:
            linked_tests.add(t)
            r = results[t]
            rows.append({**base, "test_id": t, "test_kind": r["kind"], "result": r["result"], "verdict": r["verdict"],
                         "evidence": r["evidence"], "data_origin": "SYNTHETIC"})
        for ut in ut_by_req.get(req.rid, []):
            rows.append({**base, "test_id": ut, "test_kind": "单元测试（pytest）", "result": "断言见测试代码",
                         "verdict": junit.get(ut, "未运行"), "evidence": ut.split("::")[0].replace("UT:", "tests/"),
                         "data_origin": "SYNTHETIC"})
        if req.na:   # explicit strings: pandas would read a bare "N/A" cell as NaN
            rows.append({**base, "test_id": f"N/A:{req.rid}", "test_kind": "本项目不验证", "result": req.na,
                         "verdict": "不验证（N/A）", "evidence": "需求规格.md", "data_origin": "-"})
        if not tests and not ut_by_req.get(req.rid) and not req.na:
            problems.append(f"需求没有测试也没有 N/A 理由：{req.rid}")
    for c in sorted(keys - cited_clauses):
        problems.append(f"条款没有被任何需求引用：{c}")
    for t in sorted(set(results) - linked_tests):
        problems.append(f"测试结果没有追溯到需求：{t}")
    for d in sorted(das - cited_das):
        problems.append(f"设计假设没有被任何需求引用：{d}")
    for p in parameter_rows():
        if p["provenance"] in ("?", ""):
            problems.append(f"参数没有出处：{p['parameter']}")
    if write:
        cols = ["req_id", "requirement", "kind", "source_clauses", "design_assumptions", "test_id", "test_kind",
                "result", "verdict", "evidence", "data_origin"]
        with open(OUT, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        _summary(rows, problems, keys, results, uts)
    return problems


def _summary(rows, problems, keys, results, uts) -> None:
    df = pd.DataFrame(rows)
    lines = ["<!-- generated by dms/trace.py -->", "", "| 需求 | 条款数 | 测试数 | 判定计数 |", "|---|---:|---:|---|"]
    for req in REQS:
        g = df[df.req_id == req.rid]
        cnt = g.verdict.value_counts().to_dict()
        lines.append(f"| {req.rid} {req.title} | {len(req.clauses)} | {int((~g.test_id.str.startswith('N/A:')).sum())} | "
                     + "，".join(f"{k} {v}" for k, v in cnt.items()) + " |")
    lines += ["", f"条款 {len(keys)} 条；需求 {len(REQS)} 条；场景/研究结果 {len(results)} 个；单元测试 {len(uts)} 个；"
                  f"追溯矩阵 {len(df)} 行。",
              "完整性检查：" + ("通过（条款→需求→测试与测试→需求→条款两个方向均无孤立项）" if not problems
                           else f"未通过，{len(problems)} 个问题：" + "；".join(problems[:20]))]
    (RES / "traceability_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
