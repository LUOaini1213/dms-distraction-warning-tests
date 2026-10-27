"""Traceability: clause <-> requirement <-> test in both directions, parameters with provenance, docs vs registry."""
import re
from pathlib import Path

import pandas as pd
import pytest

from dms import trace
from dms.config import parameter_rows
from dms.requirements import REQS

ROOT = Path(__file__).resolve().parents[1]


def test_registry_clauses_and_assumptions():
    """Every cited clause exists in regulation_requirements.csv and every row there is cited; DA tags exist;
    every unit test names existing requirements.  REQ: DOC-01"""
    keys = trace.clause_keys()
    cited = {c for r in REQS for c in r.clauses}
    assert cited <= keys, sorted(cited - keys)
    assert keys <= cited, sorted(keys - cited)
    assert {d for r in REQS for d in r.da} <= trace.da_tags()
    ids = {r.rid for r in REQS}
    uts = trace.unit_tests()
    assert uts and all(v and set(v) <= ids for v in uts.values()), {k: v for k, v in uts.items() if not v or not set(v) <= ids}
    assert len(ids) == len(REQS)


def test_parameters_have_provenance():
    """Every DMS parameter is traced to a clause or labelled a design assumption (DA-xx).  REQ: DOC-01"""
    rows = parameter_rows()
    assert all(r["provenance"] not in ("", "?") for r in rows)
    assert all(("DA-" in r["provenance"]) == r["kind"].startswith("设计假设") for r in rows if r["kind"] != "-")


def test_spec_document_matches_registry():
    """需求规格.md lists exactly the requirement IDs of the registry.  REQ: DOC-01"""
    spec = ROOT / "需求规格.md"
    if not spec.exists():
        pytest.skip("需求规格.md not written yet")
    text = spec.read_text(encoding="utf-8")
    in_doc = set(re.findall(r"\b((?:APP|SYS|DIS|DRW|AVL|HMI|TST|HMT|DOC)-\d{2})\b", text))
    assert in_doc == {r.rid for r in REQS}, (sorted(in_doc - {r.rid for r in REQS}), sorted({r.rid for r in REQS} - in_doc))


def test_traceability_complete_both_directions():
    """Rebuilt from results/: no orphan clause, requirement, test or design assumption.  REQ: DOC-01"""
    if not (ROOT / "results" / "regulation_tests.csv").exists():
        pytest.skip("run scripts/run_all.py first")
    problems = trace.build(write=False)
    assert problems == [], problems
    d = pd.read_csv(ROOT / "traceability.csv", encoding="utf-8-sig")
    assert set(d.req_id) == {r.rid for r in REQS}
    assert d["data_origin"].isin(["SYNTHETIC", "-"]).all()
