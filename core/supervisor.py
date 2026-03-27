import json
import multiprocessing as mp
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.llm_client import call_model
from utils.tool_schemas import LIBRARY_PARSE_TOOL, LIBRARY_SEARCH_TOOL, PRIOR_SEARCH_TOOL

from LANDAU.library import LibraryRetriever

try:
    from LANDAU.prior.prior_retrive import PriorRetriever
except Exception:  # pragma: no cover - optional dependency (faiss)
    PriorRetriever = None

try:
    from .theoretician import run_theo_node
except Exception:  # pragma: no cover - optional dependency
    run_theo_node = None

_GLOBAL_POOL: ProcessPoolExecutor | None = None


def _init_worker():
    global _WORKER_INIT
    if "_WORKER_INIT" in globals():
        return
    _WORKER_INIT = True


class SupervisorOrchestrator:
    """Single-branch linear pipeline supervisor.

    Flow per subtask:
        theo (draft) -> critic -> if complete: next subtask
                                  if to_revise/to_redraft: revise/redraft -> critic -> ...
    Repeat until all subtasks complete or max_rounds exhausted.
    """

    def __init__(
        self,
        structured_problem,
        task_dir: str,
        processes: int = 2,
        max_rounds: int = 8,
        prompts_path: str = "prompts/",
        landau_library_enabled: bool = True,
        landau_prior_enabled: bool = True,
    ):
        self.structured_problem = structured_problem
        self.task_dir = task_dir
        self.processes = max(1, int(processes))
        self.max_rounds = max(1, int(max_rounds))
        self.prompts_path = Path(prompts_path)
        self.landau_library_enabled = bool(landau_library_enabled)
        self.landau_prior_enabled = bool(landau_prior_enabled)
        if self.landau_prior_enabled and PriorRetriever is None:
            print("[Supervisor] prior retriever unavailable; disable LANDAU prior search.")
            self.landau_prior_enabled = False

        self._prior_retriever: Optional[PriorRetriever] = None
        self._library_retriever: Optional[LibraryRetriever] = None
        self.kb_search_tools: List[Dict[str, Any]] = []
        if self.landau_library_enabled:
            self.kb_search_tools.extend([LIBRARY_SEARCH_TOOL, LIBRARY_PARSE_TOOL])
        if self.landau_prior_enabled:
            self.kb_search_tools.append(PRIOR_SEARCH_TOOL)

        prompt_files = {
            "critic_prompt": "critic_prompt.txt",
            "critic_system_prompt": "critic_system_prompt.txt",
            "supervisor_prompt": "supervisor_prompt.txt",
            "supervisor_system_prompt": "supervisor_system_prompt.txt",
            "theoretician_prompt": "theoretician_prompt.txt",
            "theoretician_system_prompt": "theoretician_system_prompt.txt",
        }
        for attr, filename in prompt_files.items():
            setattr(self, attr, self._load_prompt(filename))

        self.subtasks = self._build_subtasks()

        self.node_id_counter = 1
        self.round_counter = 0

        # Trajectory: stores all nodes in order (single branch)
        self.trajectory: List[Dict[str, Any]] = []
        # Memory: accumulates critic summaries for path context
        self.path_memory: str = ""

        global _GLOBAL_POOL
        if _GLOBAL_POOL is None:
            mp.set_start_method("spawn", force=True)
            _GLOBAL_POOL = ProcessPoolExecutor(
                max_workers=self.processes,
                initializer=_init_worker,
            )

    def _load_prompt(self, filename: str) -> str:
        path = self.prompts_path / filename
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    # ── Main loop ──────────────────────────────────────────────────────
    def run(self) -> Dict[str, Any]:
        subtask_idx = 0

        while subtask_idx < len(self.subtasks) and self.round_counter < self.max_rounds:
            subtask = self.subtasks[subtask_idx]
            subtask_id = int(subtask["id"])
            description = subtask["description"]

            # ── First draft ──
            node_type = "draft"
            augmented_description = self._call_supervisor(
                subtask, node_type, description
            )

            result, node_record = self._run_theo_node(
                subtask=subtask,
                node_type=node_type,
                description=augmented_description,
            )
            self.round_counter += 1

            evaluation = self._call_critic(result)
            node_record["evaluation"] = evaluation
            node_record["reward"] = self._extract_reward(evaluation)
            node_record["memory"] = self._to_natural_text(evaluation.get("summary", ""))
            self._append_memory(evaluation)
            self.trajectory.append(node_record)

            print(
                f"[Critic] "
                f"(node_id={node_record['node_id']} subtask_id={subtask_id} node_type={node_type}) "
                f"evaluation completed decision={evaluation.get('decision', '')} "
                f"reward={node_record['reward']} 🧪"
            )

            # ── Revise/redraft loop ──
            while (
                evaluation.get("decision") != "complete"
                and self.round_counter < self.max_rounds
            ):
                decision = evaluation.get("decision", "to_revise")
                node_type = "draft" if decision == "to_redraft" else "revise"

                augmented_description = self._call_supervisor(
                    subtask, node_type, description, last_evaluation=evaluation
                )

                result, node_record = self._run_theo_node(
                    subtask=subtask,
                    node_type=node_type,
                    description=augmented_description,
                )
                self.round_counter += 1

                evaluation = self._call_critic(result)
                node_record["evaluation"] = evaluation
                node_record["reward"] = self._extract_reward(evaluation)
                node_record["memory"] = self._to_natural_text(evaluation.get("summary", ""))
                self._append_memory(evaluation)
                self.trajectory.append(node_record)

                print(
                    f"[Critic] "
                    f"(node_id={node_record['node_id']} subtask_id={subtask_id} node_type={node_type}) "
                    f"evaluation completed decision={evaluation.get('decision', '')} "
                    f"reward={node_record['reward']} 🧪"
                )

            # Move to next subtask
            subtask_idx += 1

        completed_subtasks = self._collect_completed_subtasks()

        return {
            "completed_subtasks": completed_subtasks,
            "total_rounds": self.round_counter,
            "trajectory": self.trajectory,
        }

    # ── Theoretician dispatch ──────────────────────────────────────────
    def _run_theo_node(
        self,
        subtask: Dict[str, Any],
        node_type: str,
        description: str,
    ) -> tuple:
        node_id = self.node_id_counter
        self.node_id_counter += 1
        subtask_id = int(subtask["id"])

        print(
            f"[Supervisor] "
            f"(node_id={node_id} subtask_id={subtask_id} node_type={node_type}) "
            f"task assigned 📖"
        )

        payload = {
            "depth": len(self.trajectory) + 1,
            "node_id": node_id,
            "node_type": node_type,
            "structured_problem": self.structured_problem,
            "subtask": {
                "id": subtask_id,
                "description": description,
                "subtask_type": subtask.get("subtask_type", "reasoning"),
                "input": subtask.get("input"),
                "expected_output": subtask.get("expected_output"),
            },
            "task_dir": self.task_dir,
            "path_memory": self.path_memory,
            "library_enabled": self.landau_library_enabled,
        }

        global _GLOBAL_POOL
        if run_theo_node is None:
            raise RuntimeError("run_theo_node is unavailable.")

        future = _GLOBAL_POOL.submit(run_theo_node, payload)
        try:
            node_output = future.result()
        except Exception as e:
            print(f"[Supervisor] (node_id={node_id}) Theoretician failed: {e}")
            node_output = {"result": {"error": str(e)}, "log_path": "", "depth": len(self.trajectory) + 1, "node_id": node_id}

        result = node_output.get("result", "")

        node_record = {
            "node_id": node_id,
            "depth": node_output.get("depth", len(self.trajectory) + 1),
            "subtask_id": subtask_id,
            "node_type": node_type,
            "subtask": subtask,
            "description": description,
            "result": result,
            "theoretician_output": result,
            "log_path": node_output.get("log_path", ""),
            "reward": 0.0,
            "memory": "",
            "evaluation": {},
        }

        return result, node_record

    # ── Supervisor (description generation) ────────────────────────────
    def _call_supervisor(
        self,
        subtask: Dict[str, Any],
        node_type: str,
        fallback_description: str,
        last_evaluation: Dict[str, Any] | None = None,
    ) -> str:
        if not self.supervisor_prompt:
            return fallback_description

        node_info = {
            "node_id": self.node_id_counter,
            "subtask_id": int(subtask["id"]),
            "node_type": node_type,
            "subtask_description": fallback_description,
            "evaluation": last_evaluation or {},
            "result": self.trajectory[-1]["result"] if self.trajectory else "",
            "path_memory": self.path_memory,
        }

        prompt = self.supervisor_prompt.format(
            structured=json.dumps(self.structured_problem, ensure_ascii=False, indent=2),
            node=json.dumps(node_info, ensure_ascii=False, indent=2),
        )

        try:
            response = call_model(
                system_prompt=self.supervisor_system_prompt,
                user_prompt=prompt,
                tools=self.kb_search_tools,
                tool_functions=self._kb_tool_functions_simple("Supervisor"),
            )
        except Exception:
            print("[Supervisor] call failed, using fallback description.")
            return fallback_description

        parsed = self._extract_json_object(response)
        if isinstance(parsed, dict):
            desc = str(
                parsed.get("description")
                or parsed.get("subtask_description")
                or ""
            ).strip()
            if desc:
                return desc

        return fallback_description

    # ── Critic ─────────────────────────────────────────────────────────
    def _call_critic(self, result_data: Any) -> Dict[str, Any]:
        node_output = self._extract_json_object(result_data) if isinstance(result_data, str) else (result_data or {})
        if not isinstance(node_output, dict):
            node_output = {}

        core_results = node_output.get("core_results") or node_output.get("core_result") or ""
        analysis = node_output.get("analysis") or ""
        code = node_output.get("code") or ""
        files = node_output.get("files") or []

        context_str = json.dumps(
            {"analysis": analysis, "code": code, "files": files},
            ensure_ascii=False,
            indent=2,
        )
        prompt = self.critic_prompt.format(result=core_results, context=context_str)

        response = call_model(
            system_prompt=self.critic_system_prompt,
            user_prompt=prompt,
            tools=self.kb_search_tools,
            tool_functions=self._kb_tool_functions_simple("Critic"),
        )

        parsed = self._extract_json_object(response)
        if not isinstance(parsed, dict):
            parsed = {}

        decision = str(parsed.get("decision", "to_revise")).strip().lower() or "to_revise"
        if decision not in {"to_revise", "to_redraft", "complete"}:
            decision = "to_revise"
        verdict = str(parsed.get("verdict", "")).strip().lower()
        if not verdict:
            verdict = {
                "complete": "accept",
                "to_revise": "refine",
                "to_redraft": "reject",
            }.get(decision, "refine")

        reward = self._extract_reward(parsed)
        summary = self._to_natural_text(parsed.get("summary"))
        opinion = self._to_natural_text(parsed.get("opinion"))

        return {
            "decision": decision,
            "verdict": verdict,
            "reward": reward,
            "summary": summary,
            "opinion": opinion,
            "analysis": self._to_natural_text(parsed.get("analysis") or summary or opinion),
            "code": code,
        }

    # ── Memory ─────────────────────────────────────────────────────────
    def _append_memory(self, evaluation: Dict[str, Any]):
        summary = self._to_natural_text(evaluation.get("summary", ""))
        if summary:
            self.path_memory = (self.path_memory + "\n" + summary).strip()

    # ── LANDAU Tools ───────────────────────────────────────────────────
    def _get_prior_retriever(self) -> PriorRetriever:
        if PriorRetriever is None:
            raise RuntimeError("PriorRetriever is unavailable (missing optional dependency: faiss).")
        if self._prior_retriever is None:
            self._prior_retriever = PriorRetriever()
        return self._prior_retriever

    def _get_library_retriever(self) -> LibraryRetriever:
        if self._library_retriever is None:
            self._library_retriever = LibraryRetriever()
        return self._library_retriever

    def _prior_search(
        self,
        query: str,
        top_k: int = 3,
        expand_context: bool = False,
        return_format: str = "text",
        source_ids: List[str] | None = None,
        chapter: str | None = None,
        section_prefix: str | None = None,
        keywords: List[str] | None = None,
        rewrite_query: bool = True,
    ):
        try:
            retriever = self._get_prior_retriever()
            results = retriever.retrieve(
                query=query,
                top_k=int(top_k) if top_k is not None else 3,
                expand_context=bool(expand_context),
                source_ids=source_ids,
                chapter=chapter,
                section_prefix=section_prefix,
                keywords=keywords,
                rewrite_query=bool(rewrite_query),
            )
            if return_format == "json":
                return results
            return retriever.format_for_llm(results)
        except Exception as e:
            return f"[prior_search] failed: {e}"

    def _library_search(self, query: str, top_k: int = 5):
        try:
            retriever = self._get_library_retriever()
            results = retriever.search(query=query, top_k=int(top_k) if top_k is not None else 5)
            return retriever.format_for_llm(results)
        except Exception as e:
            return f"[library_search] failed: {e}"

    def _library_parse(self, link: str, user_prompt: str, llm: str | None = None):
        try:
            retriever = self._get_library_retriever()
            results = retriever.parse(link=link, user_prompt=user_prompt, llm=llm)
            return retriever.format_parsed_for_llm(results)
        except Exception as e:
            return f"[library_parse] failed: {e}"

    def _kb_tool_functions_simple(self, agent_label: str) -> Dict[str, Any]:
        functions: Dict[str, Any] = {}
        if self.landau_library_enabled:
            functions["library_search"] = lambda **kwargs: (
                print(f"[{agent_label}] tool call library_search 🛠️"),
                self._library_search(**kwargs),
            )[1]
            functions["library_parse"] = lambda **kwargs: (
                print(f"[{agent_label}] tool call library_parse 🛠️"),
                self._library_parse(**kwargs),
            )[1]
        if self.landau_prior_enabled:
            functions["prior_search"] = lambda **kwargs: (
                print(f"[{agent_label}] tool call prior_search 🛠️"),
                self._prior_search(**kwargs),
            )[1]
        return functions

    # ── Subtask management ─────────────────────────────────────────────
    def _build_subtasks(self) -> List[Dict[str, Any]]:
        subtasks_payload = (
            self.structured_problem.get("sub-tasks")
            or self.structured_problem.get("sub_tasks")
            or self.structured_problem.get("subtasks")
            or []
        )
        if isinstance(subtasks_payload, dict):
            subtasks_payload = list(subtasks_payload.values())
        elif isinstance(subtasks_payload, str):
            subtasks_payload = [subtasks_payload]
        elif not isinstance(subtasks_payload, list):
            subtasks_payload = []

        if not subtasks_payload:
            subtasks_payload = [
                {
                    "id": 1,
                    "description": self.structured_problem.get("task_description", ""),
                    "subtask_type": "reasoning",
                    "input": self.structured_problem.get("input", ""),
                    "expected_output": self.structured_problem.get("expected_output", ""),
                }
            ]

        normalized: List[Dict[str, Any]] = []
        for item in subtasks_payload:
            if isinstance(item, dict):
                description = str(
                    item.get("description")
                    or item.get("objective")
                    or item.get("task")
                    or item.get("name")
                    or ""
                ).strip()
                sid = self._to_int(item.get("id"))
                normalized.append(
                    {
                        "id": sid,
                        "subtask_type": str(item.get("subtask_type", "reasoning")).strip() or "reasoning",
                        "input": item.get("input", self.structured_problem.get("input", "")),
                        "expected_output": item.get(
                            "expected_output",
                            self.structured_problem.get("expected_output", ""),
                        ),
                        "description": description,
                    }
                )
            else:
                text = str(item).strip()
                normalized.append(
                    {
                        "id": None,
                        "subtask_type": "reasoning",
                        "input": self.structured_problem.get("input", ""),
                        "expected_output": self.structured_problem.get("expected_output", ""),
                        "description": text,
                    }
                )

        used_ids = set()
        next_id = 1
        for item in normalized:
            sid = item.get("id")
            if sid is None or sid in used_ids:
                while next_id in used_ids:
                    next_id += 1
                sid = next_id
            used_ids.add(sid)
            next_id = max(next_id, sid + 1)
            item["id"] = sid
            if not str(item.get("description", "")).strip():
                item["description"] = f"Subtask {sid}"

        normalized.sort(key=lambda x: int(x.get("id", 0)))
        return normalized

    def _collect_completed_subtasks(self) -> List[Dict[str, Any]]:
        best_by_subtask: Dict[int, Dict[str, Any]] = {}
        for node in self.trajectory:
            evaluation = node.get("evaluation", {})
            if evaluation.get("decision") != "complete":
                continue
            sid = int(node["subtask_id"])
            prev = best_by_subtask.get(sid)
            if prev is None or float(node.get("reward", 0)) > float(prev.get("reward", 0)):
                best_by_subtask[sid] = node

        completed: List[Dict[str, Any]] = []
        for sid in sorted(best_by_subtask.keys()):
            node = best_by_subtask[sid]
            completed.append(
                {
                    "subtask_id": node["subtask_id"],
                    "description": node["description"],
                    "result": node["result"],
                    "reward": node["reward"],
                    "log_path": node.get("log_path", ""),
                }
            )
        return completed

    # ── Helpers ─────────────────────────────────────────────────────────
    def _extract_reward(self, payload: Dict[str, Any]) -> float:
        if not isinstance(payload, dict):
            return 0.0
        value = payload.get("reward")
        try:
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _to_natural_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return " ".join(self._to_natural_text(v) for v in value if self._to_natural_text(v)).strip()
        if isinstance(value, dict):
            parts = []
            for k, v in value.items():
                t = self._to_natural_text(v)
                if t:
                    parts.append(f"{k}: {t}")
            return "; ".join(parts).strip()
        return str(value).strip()

    def _extract_json_object(self, text: Any) -> Any:
        if text is None:
            return {}
        if isinstance(text, (dict, list)):
            return text

        content = str(text).strip()
        if not content:
            return {}

        try:
            return json.loads(content)
        except Exception:
            pass

        blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", content)
        for block in blocks:
            try:
                return json.loads(block.strip())
            except Exception:
                continue

        left = content.find("{")
        right = content.rfind("}")
        if left != -1 and right != -1 and right > left:
            try:
                return json.loads(content[left : right + 1])
            except Exception:
                pass

        left = content.find("[")
        right = content.rfind("]")
        if left != -1 and right != -1 and right > left:
            try:
                return json.loads(content[left : right + 1])
            except Exception:
                pass

        return {}

    def _to_int(self, value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(value)
        except Exception:
            return None
