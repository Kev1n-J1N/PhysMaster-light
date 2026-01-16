import json
from pathlib import Path
from typing import Dict, Any, List

from utils.gpt5_utils import call_model
from utils.python_utils import run_python_code
from utils.julia_env.execute import run_julia_code
from utils.skill_loader import load_skill_specs
from utils.save_utils import MarkdownWriter


PYTHON_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "Python_code_interpreter",
            "description": "Execute python code and return the stdout/stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python script to execute.",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

JULIA_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "Julia_code_interpreter",
            "description": "Execute julia code and return the result/output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "julia code to execute",
                    }
                },
                "required": ["code"],
            },
        },
    }
]

SKILL_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "load_skill_specs",
            "description": (
                "Load full specs for multiple skills by skill_ids. "
                "The returned text provides authoritative procedural guidance and should be treated as the primary source of truth when executing the corresponding skill(s). "
                "Returns plain text that contains a [SKILL SPECS] header and one or more <SKILL_SPEC> blocks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Non-empty list of skill_ids (strings), e.g. ['lamet_asymptotic_expansion', 'lamet_matching']."
                    }
                },
                "required": ["skill_ids"],
                "additionalProperties": False
            }
        }
    }
]

class Theoretician:
    def __init__(self, prompts_path: str = "prompts/",):
        self.prompts_path = Path(prompts_path)
        prompt_files = {
            "theoretician_prompt": "theoretician_prompt.txt",
            "theoretician_system_prompt": "theoretician_system_prompt.txt",
            "skills_manifest_prompt": "skills_manifest_prompt.txt"
        }
        for attr, filename in prompt_files.items():
            setattr(self, attr, self._load_prompt(filename))
        self.prompt_template = self.theoretician_prompt

    def _load_prompt(self, filename: str) -> str:
        path = self.prompts_path / filename
        with open(path, "r", encoding="utf-8") as f:
            return f.read()


    def solve(
        self,
        subtask_description: str,
        parent_memory: str | None = None,
        node_metadata: Dict[str, Any] | None = None,
        markdown_writer: MarkdownWriter | None = None,
    ) -> Dict[str, Any]:

        output_dir = str(node_metadata.get("output_dir", "")) if node_metadata else ""

        prompt = self.prompt_template.format(
            subtask=subtask_description,
            memory=(parent_memory or ""),
            node_metadata=json.dumps(node_metadata or {}, ensure_ascii=False),
            path=output_dir,
        )

        if markdown_writer:
            markdown_writer.write_to_markdown(prompt + "\n", mode="scheduler")
        print("========== Supervisor-Scheduler ========== \n" + prompt + "\n")

        tools = PYTHON_TOOL + JULIA_TOOL + SKILL_TOOL
        tool_functions = {
            "Python_code_interpreter": run_python_code,
            "Julia_code_interpreter": run_julia_code,
            "load_skill_specs": load_skill_specs,
        }

        system_prompt = self.theoretician_system_prompt

        prompt = self.skills_manifest_prompt + "\n\n" + prompt

        response = call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            tools=tools,
            tool_functions=tool_functions,
            markdown_writer=markdown_writer,
            agent_label="Theoretician",
        )

        if markdown_writer:
            markdown_writer.write_to_markdown( response + "\n", mode='theoretician_response')
        print("========== Theoretician: Response ==========\n" + response + "\n")

        return response


def run_theo_node(payload: Dict[str, Any]) -> Dict[str, Any]:

        depth = payload["depth"]
        node_index = payload["node_index"]
        structured_problem = payload["structured_problem"]
        description = payload["subtask"]["description"]
        task_dir = payload["task_dir"]

        contract_file = Path(task_dir) / "contract.json"
        with contract_file.open(encoding="utf-8") as f:
            structured_problem = json.load(f)

        raw_name = str(
            structured_problem.get("instruction_filename")
            or structured_problem.get("topic")
        )
        task_name = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in str(raw_name)
        )

        markdown_writer = MarkdownWriter(
            problem=structured_problem.get("description", ""),
            topic=structured_problem.get("topic", ""),
            log_dir=task_dir,
            depth=depth,
            node_index=node_index,
            file_prefix=task_name,
        )

        if markdown_writer:
            markdown_writer.write_to_markdown(description + "\n", mode="supervisor_scheduler")

        theoretician = Theoretician()

        node_output_dir = markdown_writer.log_dir

        node_metadata = {
            "depth": depth,
            "node_index": node_index,
            "node_type": payload["node_type"],
            "task_dir": str(task_dir),
            "output_dir": str(node_output_dir),
        }

        result = theoretician.solve(
            subtask_description=description,
            parent_memory=payload.get("parent_memory", ""),
            node_metadata=node_metadata,
            markdown_writer=markdown_writer,
        )

        return {
            "result": result,
            "log_path": markdown_writer.markdown_file,
            "depth": depth,
            "node_index": node_index,
        }
