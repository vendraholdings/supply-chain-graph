from scgraph.privacy import scan_text, ALLOWED_EMAIL


def test_flags_private_ip_and_home_path():
    assert scan_text("host 100.101.2.3 here")
    assert scan_text("/Users/someone/project/file")
    assert scan_text("ssh://box.example.ts.net")


def test_allows_noreply_address_only():
    assert not scan_text(f"User-Agent: x {ALLOWED_EMAIL}")
    assert scan_text("contact someone@example.com")
