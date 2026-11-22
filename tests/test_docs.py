"""README and paper are rendered from the same facts: every README number must appear in the paper."""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scgraph.facts import facts, fmt  # noqa: E402


def test_every_readme_fact_is_in_the_paper():
    f = facts()
    readme = (ROOT / "README.md.tmpl").read_text()
    paper = fmt((ROOT / "paper" / "paper.typ.tmpl").read_text(), f)
    missing = []
    for key in sorted(set(re.findall(r"<<\s*([A-Za-z0-9_]+)\s*>>", readme))):
        if f[key] not in paper:
            missing.append((key, f[key]))
    assert not missing, missing


def test_templates_have_no_unknown_keys():
    for p in (ROOT / "README.md.tmpl", ROOT / "paper" / "paper.typ.tmpl"):
        fmt(p.read_text())


def test_paper_compiles_to_at_most_eight_pages(tmp_path):
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_paper.py")], check=True, capture_output=True)
    import typst
    pages = typst.compile(str(ROOT / "paper" / "paper.typ"), format="png", ppi=20)
    assert 6 <= len(pages) <= 8
