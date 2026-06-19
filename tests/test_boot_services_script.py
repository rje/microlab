from __future__ import annotations

from pathlib import Path


def test_boot_services_script_covers_user_service_linger_and_nginx():
    script = Path(__file__).resolve().parents[1] / "scripts" / "setup_boot_services.sh"

    text = script.read_text(encoding="utf-8")

    assert "microlab-site.service" in text
    assert "systemctl --user enable --now microlab-site" in text
    assert "loginctl enable-linger" in text
    assert "systemctl enable --now nginx" in text
    assert "--dry-run" in text
