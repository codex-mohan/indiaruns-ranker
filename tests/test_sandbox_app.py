import os
import sys

from sandbox import app as sandbox_app
from sandbox.app import _run_command


def test_run_command_replaces_invalid_child_output_bytes():
    output = _run_command(
        [
            sys.executable,
            "-c",
            "import sys; sys.stdout.buffer.write(b'progress\\x8f done\\n')",
        ],
        env=os.environ.copy(),
        timeout=10,
        label="decode-smoke",
    )

    assert "progress" in output
    assert "done" in output


def test_launch_app_does_not_pass_theme_or_css_to_launch(monkeypatch):
    captured = {}

    def fake_launch(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(sandbox_app.app, "launch", fake_launch)

    sandbox_app.launch_app()

    assert captured == {
        "server_name": "0.0.0.0",
        "server_port": 7860,
    }


def test_gradio_api_info_schema_builds():
    info = sandbox_app.app.get_api_info()

    assert "named_endpoints" in info
