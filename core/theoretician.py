import json
import os
from pathlib import Path
from typing import Dict, Any, List

from LANDAU.library import LibraryRetriever
from utils.llm_client import call_model
from utils.python_utils import run_python_code
from utils.skill_loader import build_skill_brief_prompt, load_skill_specs
from utils.tool_schemas import LIBRARY_TOOLS, THEORETICIAN_CORE_TOOLS

class Theoretician:
    def __init__(self, prompts_path: str = "prompts/", library_enabled: bool = True):
        self.prompts_path = Path(prompts_path)
        self.library_enabled = bool(library_enabled)
        self.library_retriever = LibraryRetriever() if self.library_enabled else None
        prompt_files = {
            "theoretician_prompt": "theoretician_prompt.txt",
            "theoretician_system_prompt": "theoretician_system_prompt.txt",
        }
        for attr, filename in prompt_files.items():
            setattr(self, attr, self._load_prompt(filename))
        self.prompt_template = self.theoretician_prompt

    def _load_prompt(self, filename: str) -> str:
        path = self.prompts_path / filename
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def _library_search(self, query: str, top_k: int = 5):
        if self.library_retriever is None:
            return "[library_search] disabled"
        results = self.library_retriever.search(query=query, top_k=int(top_k) if top_k is not None else 5)
        return self.library_retriever.format_for_llm(results)

    def _library_parse(self, link: str, user_prompt: str, llm: str | None = None):
        if self.library_retriever is None:
            return "[library_parse] disabled"
        results = self.library_retriever.parse(link=link, user_prompt=user_prompt, llm=llm)
        return self.library_retriever.format_parsed_for_llm(results)

    def _log_tool_call(self, tool_name: str, node_metadata: Dict[str, Any] | None):
        node_id = node_metadata.get("node_id", "") if node_metadata else ""
        subtask_id = node_metadata.get("subtask_id", "") if node_metadata else ""
        node_type = node_metadata.get("node_type", "") if node_metadata else ""
        print(
            f"[Theoretician] "
            f"(node_id={node_id} subtask_id={subtask_id} node_type={node_type}) "
            f"tool call {tool_name} 🛠️"
        )


    def solve(
        self,
        subtask_description: str,
        path_memory: str | None = None,
        node_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:

        output_dir = str(node_metadata.get("output_dir", "")) if node_metadata else ""

        prompt = self.prompt_template.format(
            subtask=subtask_description,
            memory=(path_memory or ""),
            node_metadata=json.dumps(node_metadata or {}, ensure_ascii=False),
            path=output_dir,
        )

        tools = THEORETICIAN_CORE_TOOLS + (LIBRARY_TOOLS if self.library_enabled else [])
        tool_functions = {
            "Python_code_interpreter": run_python_code,
            "load_skill_specs": load_skill_specs,
        }
        if self.library_enabled:
            tool_functions["library_search"] = self._library_search
            tool_functions["library_parse"] = self._library_parse

        system_prompt = self.theoretician_system_prompt
        wrapped_tool_functions = {
            "Python_code_interpreter": lambda **kwargs: (
                self._log_tool_call("Python_code_interpreter", node_metadata),
                run_python_code(**kwargs),
            )[1],
            "load_skill_specs": lambda **kwargs: (
                self._log_tool_call("load_skill_specs", node_metadata),
                load_skill_specs(**kwargs),
            )[1],
        }
        if self.library_enabled:
            wrapped_tool_functions["library_search"] = lambda **kwargs: (
                self._log_tool_call("library_search", node_metadata),
                self._library_search(**kwargs),
            )[1]
            wrapped_tool_functions["library_parse"] = lambda **kwargs: (
                self._log_tool_call("library_parse", node_metadata),
                self._library_parse(**kwargs),
            )[1]

        prompt = build_skill_brief_prompt() + "\n\n" + prompt

        response = call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            tools=tools,
            tool_functions=wrapped_tool_functions,
        )

        return response


def run_theo_node(payload: Dict[str, Any]) -> Dict[str, Any]:

        depth = payload["depth"]
        node_id = int(payload["node_id"])
        structured_problem = payload["structured_problem"]
        description = payload["subtask"]["description"]
        task_dir = payload["task_dir"]

        contract_file = Path(task_dir) / "contract.json"
        with contract_file.open(encoding="utf-8") as f:
            structured_problem = json.load(f)

        theoretician = Theoretician(library_enabled=bool(payload.get("library_enabled", True)))

        node_output_dir = Path(task_dir) / f"node_{node_id}"
        os.makedirs(node_output_dir, exist_ok=True)

        node_metadata = {
            "depth": depth,
            "node_id": node_id,
            "subtask_id": payload["subtask"]["id"],
            "node_type": payload["node_type"],
            "task_dir": str(task_dir),
            "output_dir": str(node_output_dir),
        }

        result = theoretician.solve(
            subtask_description=description,
            path_memory=payload.get("path_memory", payload.get("parent_memory", "")),
            node_metadata=node_metadata,
        )
        print(
            f"[Theoretician] "
            f"(node_id={node_id} subtask_id={payload['subtask']['id']} node_type={payload['node_type']}) "
            f"task completed ✅"
        )

        return {
            "result": result,
            "log_path": str(node_output_dir),
            "depth": depth,
            "node_id": node_id,
        }
