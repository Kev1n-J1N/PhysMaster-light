# Librarian/librarian.py

import sys
import shutil
from pathlib import Path
from typing import Any, Dict, Optional
import subprocess


def clear_dir(path: Path) -> None:
    """
    安全清空一个目录下的所有内容（但保留目录本身）
    """
    if not path.exists():
        return
    for p in path.iterdir():
        if p.is_file():
            p.unlink()
        elif p.is_dir():
            shutil.rmtree(p)


def run_pasax_for_kb(
    task_dir: Path,
    task_name: str,
    kb_cfg: Dict[str, Any],
) -> Optional[Path]:
    """
    1. 用 Clarifier 生成的 contract.json 更新 PasaX 的 templete.json；
    2. 调 Pasa-X-papermaster_new/run.py，让它在自身的 librarian/ 里生成 JSON；
    3. 把 Pasa-X-papermaster_new/librarian/*.json 拷贝到
           项目根目录的 knowledge_base/local_knowledge_base/<task_name>/；
    4. 若 write_local_to_global = true，再把本次 local 结果追加写入
           knowledge_base/global_knowledge_base/ 下面；
    5. 若 cleanup_pasax_librarian = true，用完后清空 Pasa-X 里的 librarian。
    """
    if not kb_cfg.get("enabled", False):
        print("[KB] knowledge_base.enabled = false，跳过 PasaX")
        return None

    parax_cfg = kb_cfg.get("pasax", {}) or {}

    project_root = Path(__file__).resolve().parents[1]
  
    default_root = "agent/librarian/Pasa-X-papermaster_new"
    parax_root = (project_root / parax_cfg.get("root_path", default_root)).resolve()
    parax_run_py = parax_root / "run.py"

    if not parax_run_py.exists():
        print(f"[Main-KB] ParaX run.py 未找到：{parax_run_py}，跳过 PasaX")
        return None

    structured_path = task_dir / "contract.json"
    if not structured_path.exists():
        print(f"[Main-KB] contract.json 不存在：{structured_path}，跳过 PasaX")
        return None
    
    parax_inputs = parax_root / "inputs&transfer"
    parax_inputs.mkdir(parents=True, exist_ok=True)

    templete_json = parax_inputs / "templete.json"
    shutil.copy2(structured_path, templete_json)
    print(f"[Main-KB] 已将 contract.json 拷贝到 ParaX: {templete_json}")

    parax_librarian_dir = parax_root / "librarian"
    parax_librarian_dir.mkdir(parents=True, exist_ok=True)

    if kb_cfg.get("cleanup_parax_librarian", False):
        print(f"[Main-KB] cleanup_parax_librarian = True，先清空 ParaX librarian: {parax_librarian_dir}")
        clear_dir(parax_librarian_dir)

    base_local_root = kb_cfg.get("local_root", "knowledge_base/local_knowledge_base")
    task_local_kb_dir = (project_root / base_local_root / task_name).resolve()
    task_local_kb_dir.mkdir(parents=True, exist_ok=True)

    clear_dir(task_local_kb_dir)
    print(f"[Main-KB] 本次任务的 local KB 目录: {task_local_kb_dir}")

    cmd = [
        sys.executable,
        str(parax_run_py),
    ]

    print("[Main-KB] 调用 PasaX 构建本地知识库：")
    print("         ", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True, cwd=str(parax_root))
        print(f"[Main-KB] PasaX 完成。结果写入: {parax_librarian_dir}")
    except subprocess.CalledProcessError as e:
        print(f"[Main-KB] PasaX 运行失败，忽略本次 KB 构建: {e}")
        return None

    json_files = sorted(parax_librarian_dir.glob("*.json"))
    if not json_files:
        print(f"[Main-KB] PasaX librarian 目录下没有找到任何 JSON 文件: {parax_librarian_dir}")
    else:
        for src in json_files:
            dst = task_local_kb_dir / src.name
            shutil.copy2(src, dst)
        print(f"[Main-KB] 已从 ParaX librarian 拷贝 {len(json_files)} 个 JSON 到 local KB: {task_local_kb_dir}")

    if kb_cfg.get("write_local_to_global", False):
        global_root = kb_cfg.get("global_root", "knowledge_base/global_knowledge_base")
        global_root_dir = (project_root / global_root).resolve()
        global_root_dir.mkdir(parents=True, exist_ok=True)

        num_copied = 0
        for src in task_local_kb_dir.glob("*.json"):
            dst_name = f"{task_name}__{src.name}"
            dst = global_root_dir / dst_name
            shutil.copy2(src, dst)
            num_copied += 1

        print(f"[Main-KB] 已将本次 local KB 中的 {num_copied} 个 JSON 追加写入 global KB: {global_root_dir}")

    if kb_cfg.get("cleanup_parax_librarian", False):
        print(f"[Main-KB] cleanup_parax_librarian = True，任务结束清空 ParaX librarian: {parax_librarian_dir}")
        clear_dir(parax_librarian_dir)

    return task_local_kb_dir
