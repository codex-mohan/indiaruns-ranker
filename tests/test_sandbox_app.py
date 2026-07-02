import os
import sys

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
