"""The ffmpeg harness preserves process arguments and the selected interpreter."""

import json
import subprocess
import sys

from datasets.vpt.video.fake_program import program


def test_launcher_preserves_arguments_and_exit_code(tmp_path):
    script, control, log = (
        tmp_path / name for name in ("fake code.py", "control 'x'", "log")
    )
    script.write_text(
        "import json, pathlib, sys\n"
        "data = {'argv': sys.argv[1:], 'python': sys.executable, "
        "'isolated': sys.flags.isolated, 'no_site': sys.flags.no_site}\n"
        "pathlib.Path(sys.argv[2]).write_text(json.dumps(data))\n"
        "sys.exit(7)\n"
    )
    arguments = ["-i", "video with spaces", "quote'\"", "line\nbreak"]
    for name in ("ffmpeg one", "ffmpeg two"):
        executable = tmp_path / name
        program(executable, script, control, log)
        result = subprocess.run([str(executable), *arguments], check=False, timeout=3)
        assert result.returncode == 7
        assert json.loads(log.read_text()) == {
            "argv": [str(control), str(log), str(executable), *arguments],
            "python": sys.executable,
            "isolated": 1,
            "no_site": 1,
        }
