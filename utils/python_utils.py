import os
import subprocess
import tempfile

def run_python_code(code: str) -> str:
    """
    在本地 Python 解释器中执行 code，返回 stdout+stderr。
    """
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        f.flush()
        tmp_path = f.name

    try:
        completed = subprocess.run(
            [subprocess.sys.executable, tmp_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1800,
        )
        return completed.stdout
    finally:
        os.unlink(tmp_path)