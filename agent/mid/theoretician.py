import json
from pathlib import Path
from typing import Dict, Any, List

from utils.gpt5_utils import call_model, execute_python_code
from utils.julia_env.execute import run_julia_code
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

class Theoretician:

    def __init__(self, prompt_path: str = "prompts/theoretician_prompt.txt"):
        self.prompt_path = Path(prompt_path)
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()

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

        tools = PYTHON_TOOL + JULIA_TOOL
        tool_functions = {
            "Python_code_interpreter": execute_python_code,
            "Julia_code_interpreter": run_julia_code,
        }

        system_prompt = (
            """
            As a theoretical and computational physicist, your core responsibilities are:
            1. Analyse the problem with scientific rigor and establish appropriate theoretical models with physical intuition.
            2. Propose numerical methods (Monte Carlo, DMRG etc.) and conduct numerical simulation with code.
            NOTICE: YOU MUST RUN THE CODE TO GET THE ANSWER!
            NOTICE: Try to use more precise method (eg. Quantum Monte Carlo) without systematic error first. Only when such attempt failed, introduce proper methods with  approximation (eg. mean-field approximations) to simplify the problem.
            NOTICE: You should always keep your code concise as far as possible.  Try to avoid excessively long and complexed code. 
            """
        )

        # log query
        if markdown_writer:
            markdown_writer.write_to_markdown( prompt + "\n", mode='theoretician_query')
        print("========== Theoretician: Query ========== \n" + prompt + "\n")

        response = call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            tools=tools,
            tool_functions=tool_functions,
            markdown_writer=markdown_writer,
        )

        if markdown_writer:
            markdown_writer.write_to_markdown( prompt + "\n", mode='theoretician_response')
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
        log_dir= task_dir,
        depth=depth,
        node_index=node_index,
        file_prefix=task_name,
    )

    if markdown_writer:
        markdown_writer.write_to_markdown( description + "\n", mode='supervisor_scheduler')    

    theoretician = Theoretician()
    node_metadata = {
        "depth": depth,
        "node_index": node_index,
        "node_type": payload["node_type"],
        "task_dir": str(task_dir),
    }

    result = theoretician.solve(
        subtask_description=description,
        parent_memory=payload.get("parent_memory", ""),
        node_metadata=node_metadata,
        markdown_writer=markdown_writer,
    )

    # writer.write_to_markdown(json.dumps(result, ensure_ascii=False, indent=2), "theoretician_result")

    return {
        "result": result,
        "log_path": markdown_writer.markdown_file,
        "depth": depth,
        "node_index": node_index,
    }

