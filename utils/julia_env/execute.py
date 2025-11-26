"""
utils.julia_env.execute

占位版 Julia 执行接口：
在没有 Julia 环境的服务器上避免 import 报错，
同时在调用时给出明确的占位说明。
"""

from typing import Optional


def run_julia_code(code: str, work_dir: Optional[str] = None) -> str:
    """
    伪实现：不真正运行 Julia，只返回一段说明文字。

    参数
    ----
    code : str
        原本要在 Julia 中执行的代码字符串。
    work_dir : Optional[str]
        原本打算作为工作目录的路径（当前实现中不使用）。

    返回
    ----
    str
        一段说明字符串，告诉上层：Julia 在当前环境中被跳过。
    """
    preview = code.strip().splitlines()
    if preview:
        preview = preview[:5]
        preview_str = "\n".join(preview)
    else:
        preview_str = "<empty code>"

    return (
        "[run_julia_code placeholder] Julia execution is skipped on this server.\n"
        "Working directory: {wd}\n"
        "Code preview (first lines):\n{preview}\n"
    ).format(
        wd=work_dir or "<current dir>",
        preview=preview_str
    )
