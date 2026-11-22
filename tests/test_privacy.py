from scgraph.privacy import scan_text, ALLOWED_EMAIL


def test_flags_private_ip_and_home_path():
    assert scan_text("host 10" + "0.101.2.3 here")
    assert scan_text("/Us" + "ers/someone/project/file")
    assert scan_text("ssh://box.example.ts" + ".net")


def test_allows_noreply_address_only():
    assert not scan_text(f"User-Agent: x {ALLOWED_EMAIL}")
    assert scan_text("contact someone" + "@example.com")


def test_no_personal_mail_address_in_tracked_files():
    import re
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    files = subprocess.run(["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True).stdout.split()
    pat = re.compile(r"@(?:gmail|googlemail|outlook|hotmail|icloud|yahoo|proton|protonmail)\b", re.I)
    bad = []
    for f in files:
        p = root / f
        if p.suffix in {".png", ".gif", ".pdf", ".parquet", ".js", ".jpg"} or not p.is_file():
            continue
        try:
            text = p.read_text()
        except UnicodeDecodeError:
            continue
        if pat.search(text):
            bad.append(f)
    assert not bad, bad
