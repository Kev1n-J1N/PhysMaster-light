import os
import time
import re
import uuid

class MarkdownWriter:
    def __init__(
        self,
        problem: str,
        topic: str,
        log_dir: str = "logs",
        depth: int | None = None,
        node_index: int | None = None,
        *,
        file_prefix: str | None = None,
        markdown_file: str | None = None,
    ):
        self.problem = problem
        self.topic = topic
        self.log_dir = log_dir
        self.depth = depth
        self.node_index = node_index
        self.file_prefix = file_prefix

        os.makedirs(self.log_dir, exist_ok=True)

        # If a specific markdown file path was provided, use it directly.
        if markdown_file:
            self.markdown_file = str(markdown_file)
        else:
            self.markdown_file = self.get_markdown_file()

        self.buffer = []

        # Only write the header if the markdown file does not already exist or is empty.
        try:
            file_exists = os.path.exists(self.markdown_file)
            file_nonempty = file_exists and os.path.getsize(self.markdown_file) > 0
        except Exception:
            file_nonempty = False

        if not file_nonempty:
            header = [
                "# Topic\n",
                f"{self.topic}\n",
                "# Task\n",
                f"{self.problem}\n"
            ]
            self._write_lines(header)

    def get_markdown_file(self):
        timestamp = time.strftime('%m%d')
        if self.file_prefix:
            raw_prefix = str(self.file_prefix)
        else:
            raw_prefix = self.problem or 'problem'

        sanitized = re.sub(r'[^A-Za-z0-9\u4e00-\u9fff_-]+', '_', raw_prefix.strip())
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        prefix = sanitized[:60] if sanitized else 'problem'

        extra_parts = []
        if self.depth is not None:
            extra_parts.append(f"depth{self.depth}")
        if self.node_index is not None:
            extra_parts.append(f"node{self.node_index}")
        extra = "_".join(extra_parts)

        if extra:
            filename = f"{timestamp}_{prefix}_{extra}.md"
        else:
            filename = f"{timestamp}_{prefix}.md"

        return os.path.join(self.log_dir, filename)

    def _write_lines(self, lines: list[str]):
        """写入文件并保存到内存 buffer"""
        with open(self.markdown_file, 'a', encoding='utf-8') as f:
            for line in lines:
                f.write(line)
                self.buffer.append(line)

    def write_to_markdown(self, text: str, mode: str = 'supervisor'):
        if text is None:
            text = ""

        if mode not in ('julia_call', 'julia_result'):
            text = text.replace("#", "\\#")  # 转义 Markdown #

        if mode == 'supervisor_scheduler':
            self._write_lines([
                "# Supervisor-Scheduler\n",
                f"{text}\n"
            ])
        elif mode == 'supervisor_critic':
            self._write_lines([
                "# Supervisor-Critic Evaluation\n",
                f"{text}\n"
            ])
        elif mode == 'theoretician_query':
            self._write_lines([
                "# Theoretician Task\n",
                f"{text}\n"
            ])
        elif mode == 'theoretician_response':
            self._write_lines([
                "# Theoretician Solution\n",
                f"{text}\n"
            ])
        elif mode == 'julia_call':
            self._write_lines([
                f"```julia\n{text}\n```\n"
            ])
        elif mode == 'julia_result':
            self._write_lines([
                f"```\n{text}\n```\n"
            ])

    def get_buffer(self) -> str:
        """返回内存中完整 Markdown 内容"""
        return "".join(self.buffer)
