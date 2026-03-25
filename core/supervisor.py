import json
import math
import multiprocessing as mp
import re
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils.llm_client import call_model
from utils.tool_schemas import LIBRARY_PARSE_TOOL, LIBRARY_SEARCH_TOOL, PRIOR_SEARCH_TOOL

from LANDAU.library import LibraryRetriever
from .mcts import MCTSNode, MCTSTree

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
    """MCTS Supervisor，负责搜索、调度、评估与轨迹汇总。"""

    def __init__(
        self,
        structured_problem,
        task_dir: str,
        processes: int = 2,
        max_rounds: int = 8,
        prompts_path: str = "prompts/",
        draft_expansion: int = 2,
        revise_expansion: int = 2,
        exploration_constant: float = 1.414,
        active_beam_width: int = 0,
        landau_library_enabled: bool = True,
        landau_prior_enabled: bool = True,
    ):
        self.structured_problem = structured_problem
        self.task_dir = task_dir
        self.processes = max(1, int(processes))
        self.max_rounds = max(1, int(max_rounds))
        self.prompts_path = Path(prompts_path)
        self.draft_expansion = max(1, int(draft_expansion))
        self.revise_expansion = max(1, int(revise_expansion))
        self.exploration_constant = float(exploration_constant)
        self.active_beam_width = max(0, int(active_beam_width or 0))
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

        self.tree = MCTSTree(
            root_subtask_id=0,
            root_description="Virtual Root",
        )
        self.tree.root.node_type = "virtual"
        self.tree.root.status = "completed_expended"
        self.tree.root.subtask_description = "Virtual Root"
        self.tree.root.subtask_payload = None
        self.tree.root.evaluation = {
            "decision": "complete",
            "reward": 0.0,
            "verdict": "accept",
            "summary": "Virtual root initialization.",
        }
        self.tree.root.supervisor_dispatch = {"node_type": "virtual", "description": "Virtual Root"}
        self.tree.root.supervisor_feedback = dict(self.tree.root.supervisor_dispatch)
        self.tree.root.visits = 1
        self.tree.root.total_reward = 0.0
        self.tree.root.average_reward = 0.0

        self.node_id_counter = 1
        self.round_counter = 0

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

    def run(self) -> Dict[str, Any]:
        while self.round_counter < self.max_rounds:
            selected_node = self._select_leaf_node()
            if selected_node is None:
                break

            dispatch = self._resolve_dispatch(selected_node)
            if dispatch.get("stop_search", False):
                break

            selected_node.selected_round = self.round_counter
            new_nodes = self._expand_and_simulate_nodes(
                parent=selected_node,
                node_type=dispatch["node_type"],
                count=dispatch["expansion_count"],
                subtask=dispatch["subtask"],
                augmented_description=dispatch["description"],
                supervisor_dispatch=dispatch["supervisor_dispatch"],
                round_index=self.round_counter,
            )
            self.round_counter += 1

            if self._find_full_completion_path() is not None:
                break

        best_path_nodes = self._find_best_path_nodes()
        completed_subtasks = self._collect_completed_subtasks()
        trajectory = self._serialize_trajectory(best_path_nodes)

        summary = {
            "completed_subtasks": completed_subtasks,
            "total_nodes": len(self.tree.get_all_nodes()),
            "total_rounds": self.round_counter,
            "tree_stats": self.tree.get_tree_stats(),
            "trajectory": trajectory,
        }
        return summary

    def _resolve_dispatch(self, node: MCTSNode) -> Dict[str, Any]:
        supervisor_raw = ""
        try:
            supervisor_raw = self._call_supervisor(node) or ""
        except Exception:
            supervisor_raw = ""
            print("[Supervisor] call failed.")

        supervisor_payload = self._extract_json_object(supervisor_raw)
        if not isinstance(supervisor_payload, dict):
            supervisor_payload = {}

        default_decision = "to_redraft" if node.node_type == "virtual" else "to_revise"
        decision = str((node.evaluation or {}).get("decision", default_decision)).strip().lower()
        default_node_type = "draft" if node.node_type == "virtual" else self._decision_to_node_type(decision)

        default_subtask_id = self._default_next_subtask_id(node, decision)
        if default_subtask_id is None:
            return {"stop_search": True}

        subtask_id = self._extract_requested_subtask_id(
            supervisor_payload,
            fallback=default_subtask_id,
        )
        subtask = self._get_subtask_by_id(subtask_id) or self._get_subtask_by_id(default_subtask_id)
        if not subtask:
            return {"stop_search": True}

        node_type = self._sanitize_node_type(supervisor_payload.get("node_type"), default_node_type)
        expansion_count = self._get_expansion_count_by_node_type(node_type)

        description = str(
            supervisor_payload.get("description")
            or supervisor_payload.get("subtask_description")
            or subtask.get("description")
            or node.subtask_description
            or ""
        ).strip() or str(subtask.get("description", "")).strip()

        supervisor_dispatch = {
            "node_type": node_type,
            "subtask_id": subtask["id"],
            "subtask": subtask,
            "description": description,
            "raw_supervisor_output": supervisor_raw,
        }
        return {
            "stop_search": False,
            "node_type": node_type,
            "expansion_count": max(1, int(expansion_count)),
            "subtask": subtask,
            "description": description,
            "supervisor_dispatch": supervisor_dispatch,
        }

    def _call_supervisor(self, node: MCTSNode) -> str:
        if not self.supervisor_prompt:
            return ""

        try:
            node_info = {
                "node_id": node.node_id,
                "subtask_id": node.subtask_id,
                "node_type": node.node_type,
                "subtask_description": node.subtask_description,
                "evaluation": node.evaluation,
                "result": node.result,
                "path_memory": self._compose_parent_memory(node),
            }
        except Exception:
            node_info = {}

        prompt = self.supervisor_prompt.format(
            structured=json.dumps(self.structured_problem, ensure_ascii=False, indent=2),
            node=json.dumps(node_info, ensure_ascii=False, indent=2),
        )

        response = call_model(
            system_prompt=self.supervisor_system_prompt,
            user_prompt=prompt,
            tools=self.kb_search_tools,
            tool_functions=self._kb_tool_functions("Supervisor", node),
            model_name="gpt-5",
        )
        return response

    def _call_critic(self, node: MCTSNode) -> Dict[str, Any]:
        result_data = node.result or ""
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
            tool_functions=self._kb_tool_functions("Critic", node),
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
        analysis_text = self._to_natural_text(parsed.get("analysis") or summary or opinion)

        return {
            "decision": decision,
            "verdict": verdict,
            "reward": reward,
            "summary": summary,
            "opinion": opinion,
            "analysis": analysis_text,
            "code": code,
        }

    def _expand_and_simulate_nodes(
        self,
        parent: MCTSNode,
        node_type: str,
        count: int,
        subtask: Dict[str, Any],
        augmented_description: str,
        supervisor_dispatch: Dict[str, Any],
        round_index: int,
    ) -> List[MCTSNode]:
        if parent and parent.status == "completed_closed":
            return []

        new_nodes: List[MCTSNode] = []
        outputs: List[Tuple[MCTSNode, Dict[str, Any]]] = []

        subtask_id = int(subtask.get("id", 1))
        subtask_type = str(subtask.get("subtask_type", "reasoning"))

        global _GLOBAL_POOL
        if run_theo_node is None:
            raise RuntimeError("run_theo_node is unavailable.")
        futures = []
        node_ids: List[int] = []

        for _ in range(max(1, int(count))):
            node_id = int(self.node_id_counter)
            self.node_id_counter += 1
            node_ids.append(node_id)

            payload = {
                "depth": parent.get_depth() + 1,
                "node_id": node_id,
                "node_type": node_type,
                "structured_problem": self.structured_problem,
                "subtask": {
                    "id": subtask_id,
                    "description": augmented_description,
                    "subtask_type": subtask_type,
                    "input": subtask.get("input"),
                    "expected_output": subtask.get("expected_output"),
                },
                "task_dir": self.task_dir,
                "path_memory": self._compose_parent_memory(parent),
                "library_enabled": self.landau_library_enabled,
            }
            print(
                f"[Supervisor] "
                f"(node_id={node_id} subtask_id={subtask_id} node_type={node_type}) "
                f"task assigned 📖"
            )
            futures.append(_GLOBAL_POOL.submit(run_theo_node, payload))

        wait(futures)

        for idx, future in enumerate(futures):
            node_id = node_ids[idx]
            child_node = MCTSNode(
                subtask_id=subtask_id,
                subtask_payload=subtask,
                node_id=node_id,
                node_type=node_type,
                subtask_description=augmented_description,
                status="open",
                created_by="supervisor",
            )
            child_node.supervisor_dispatch = dict(supervisor_dispatch)
            child_node.supervisor_feedback = dict(supervisor_dispatch)
            child_node.selected_round = round_index

            self.tree.add_node(child_node)
            parent.add_child(child_node)
            new_nodes.append(child_node)

            try:
                node_output = future.result()
            except Exception as e:
                child_node.status = "failed"
                child_node.result = {"error": str(e)}
                child_node.theoretician_output = child_node.result
                child_node.evaluation = {
                    "decision": "to_revise",
                    "verdict": "refine",
                    "reward": 0.0,
                    "summary": "Theoretician execution failed.",
                    "opinion": str(e),
                }
                child_node.reward = 0.0
                child_node.memory = self._compose_node_memory(child_node, child_node.evaluation)
                child_node.backpropagate(0.0)
                continue

            child_node.result = node_output.get("result")
            child_node.theoretician_output = node_output.get("result")
            child_node.log_path = node_output.get("log_path")
            outputs.append((child_node, node_output))

        for child_node, _ in outputs:
            evaluation = self._call_critic(child_node)
            child_node.evaluation = evaluation
            reward = self._extract_reward(evaluation)
            child_node.reward = reward
            print(
                f"[Critic] "
                f"(node_id={child_node.node_id} subtask_id={child_node.subtask_id} node_type={child_node.node_type}) "
                f"evaluation completed decision={evaluation.get('decision', '')} reward={reward} 🧪"
            )
            child_node.memory = self._compose_node_memory(child_node, evaluation)
            child_node.status = "completed"
            child_node.backpropagate(reward)

        if new_nodes:
            self._apply_beam_pruning(parent.get_depth() + 1)
            if parent.status not in {"completed_closed", "failed"}:
                parent.status = "completed_expended"

        return new_nodes

    # Tools
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

    def _log_tool_call(self, agent_label: str, node: MCTSNode, tool_name: str):
        print(
            f"[{agent_label}] "
            f"(node_id={node.node_id} subtask_id={node.subtask_id} node_type={node.node_type}) "
            f"tool call {tool_name} 🛠️"
        )

    def _kb_tool_functions(self, agent_label: str, node: MCTSNode) -> Dict[str, Any]:
        functions: Dict[str, Any] = {}
        if self.landau_library_enabled:
            functions["library_search"] = lambda **kwargs: (
                self._log_tool_call(agent_label, node, "library_search"),
                self._library_search(**kwargs),
            )[1]
            functions["library_parse"] = lambda **kwargs: (
                self._log_tool_call(agent_label, node, "library_parse"),
                self._library_parse(**kwargs),
            )[1]
        if self.landau_prior_enabled:
            functions["prior_search"] = lambda **kwargs: (
                self._log_tool_call(agent_label, node, "prior_search"),
                self._prior_search(**kwargs),
            )[1]
        return functions

    # Node selection / pruning
    def _select_leaf_node(self) -> Optional[MCTSNode]:
        if not self.tree.root.children:
            return self.tree.root

        candidates = [
            node
            for node in self.tree.get_all_nodes()
            if node.node_type != "virtual" and node.status not in {"completed_closed", "failed"}
        ]
        if not candidates:
            return None

        def ucb(node: MCTSNode) -> float:
            if node.visits <= 0:
                return float("inf")
            parent_visits = node.parent.visits if node.parent else self.tree.root.visits
            parent_visits = max(1, int(parent_visits))
            exploit = node.average_reward
            explore = self.exploration_constant * math.sqrt(math.log(parent_visits + 1) / node.visits)
            return exploit + explore

        selected = max(
            candidates,
            key=lambda n: (ucb(n), n.get_depth(), -n.node_id),
        )
        return selected

    def _apply_beam_pruning(self, depth: int):
        if self.active_beam_width <= 0:
            return

        candidates = [
            n
            for n in self.tree.get_all_nodes()
            if n.node_type != "virtual"
            and n.status not in {"completed_closed", "failed"}
            and n.get_depth() == depth
        ]
        if len(candidates) <= self.active_beam_width:
            return

        ranked = sorted(
            candidates,
            key=lambda n: (
                float(n.reward or 0.0),
                float(n.get_reward_value()),
                int(n.visits),
                -n.node_id,
            ),
            reverse=True,
        )
        keep = {n.node_id for n in ranked[: self.active_beam_width]}
        for node in candidates:
            if node.node_id not in keep:
                node.status = "completed_closed"

    # Subtask normalization
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

    def _default_next_subtask_id(self, node: MCTSNode, decision: str) -> Optional[int]:
        if not self.subtasks:
            return None
        if node.node_type == "virtual":
            return int(self.subtasks[0]["id"])

        current_subtask_id = int(node.subtask_id)
        if self._get_subtask_by_id(current_subtask_id) is None:
            current_subtask_id = int(self.subtasks[0]["id"])

        is_complete = decision == "complete" or node.is_subtask_complete()
        if is_complete:
            next_subtask = self._get_next_subtask(current_subtask_id)
            return int(next_subtask["id"]) if next_subtask else None
        return current_subtask_id

    def _extract_requested_subtask_id(self, payload: Dict[str, Any], fallback: int) -> int:
        sid = self._to_int(payload.get("subtask_id"))
        if sid is not None and self._get_subtask_by_id(sid) is not None:
            return sid

        subtask_obj = payload.get("subtask")
        if isinstance(subtask_obj, dict):
            sid = self._to_int(subtask_obj.get("id"))
            if sid is not None and self._get_subtask_by_id(sid) is not None:
                return sid
        return fallback

    def _get_subtask_by_id(self, subtask_id: int) -> Optional[Dict[str, Any]]:
        for subtask in self.subtasks:
            if int(subtask.get("id", -1)) == int(subtask_id):
                return subtask
        return None

    def _get_next_subtask(self, current_subtask_id: int) -> Optional[Dict[str, Any]]:
        for idx, subtask in enumerate(self.subtasks):
            if int(subtask.get("id", -1)) == int(current_subtask_id):
                next_idx = idx + 1
                return self.subtasks[next_idx] if next_idx < len(self.subtasks) else None
        return None

    # Best trajectory
    def _get_path_nodes(self, node: MCTSNode) -> List[MCTSNode]:
        path: List[MCTSNode] = []
        current: Optional[MCTSNode] = node
        while current is not None:
            path.append(current)
            current = current.parent
        path.reverse()
        return path

    def _path_reward_sum(self, path_nodes: List[MCTSNode]) -> float:
        return float(sum(float(n.reward or 0.0) for n in path_nodes if n.node_type != "virtual"))

    def _count_completed_subtasks_in_path(self, path_nodes: List[MCTSNode]) -> Tuple[int, set[int]]:
        completed: set[int] = set()
        for node in path_nodes:
            if node.node_type == "virtual":
                continue
            if node.is_subtask_complete():
                completed.add(int(node.subtask_id))
        return len(completed), completed

    def _find_full_completion_path(self) -> Optional[List[MCTSNode]]:
        total_subtasks = len(self.subtasks)
        if total_subtasks <= 0:
            return None

        candidates: List[List[MCTSNode]] = []
        for node in self.tree.get_all_nodes():
            if node.node_type == "virtual" or node.visits <= 0:
                continue
            path = self._get_path_nodes(node)
            completed_count, _ = self._count_completed_subtasks_in_path(path)
            if completed_count >= total_subtasks:
                candidates.append(path)

        if not candidates:
            return None
        return max(
            candidates,
            key=lambda path: (
                self._path_reward_sum(path),
                len(path),
                path[-1].visits if path else 0,
                path[-1].node_id if path else -1,
            ),
        )

    def _resolve_best_path(self) -> List[MCTSNode]:
        candidates = [n for n in self.tree.get_all_nodes() if n.node_type != "virtual" and n.visits > 0]
        if not candidates:
            return [self.tree.root]

        return max(
            (self._get_path_nodes(node) for node in candidates),
            key=lambda path: (
                self._count_completed_subtasks_in_path(path)[0],
                self._path_reward_sum(path),
                len(path),
                path[-1].get_reward_value() if path else 0.0,
                path[-1].node_id if path else -1,
            ),
        )

    def _find_best_path_nodes(self) -> List[MCTSNode]:
        full = self._find_full_completion_path()
        if full is not None:
            return full
        return self._resolve_best_path()

    def _find_best_trajectory(self) -> List[Dict[str, Any]]:
        """Backward-compatible alias: returns serialized best trajectory."""
        return self._serialize_trajectory(self._find_best_path_nodes())

    def _serialize_trajectory(self, path_nodes: List[MCTSNode]) -> List[Dict[str, Any]]:
        if not path_nodes:
            return []
        return [
            {
                "node_id": node.node_id,
                "depth": node.get_depth(),
                "subtask_id": node.subtask_id,
                "node_type": node.node_type,
                "subtask": node.subtask_payload,
                "description": node.subtask_description,
                "result": node.result,
                "reward": node.reward,
                "visits": node.visits,
                "memory": node.memory,
                "log_path": node.log_path,
                "supervisor_dispatch": node.supervisor_dispatch,
                "critic_feedback": node.evaluation,
                "theoretician_output": node.theoretician_output,
            }
            for node in path_nodes
            if node.node_type != "virtual"
        ]

    def _collect_completed_subtasks(self) -> List[Dict[str, Any]]:
        best_by_subtask: Dict[int, MCTSNode] = {}
        for node in self.tree.get_all_nodes():
            if node.node_type == "virtual" or not node.is_subtask_complete():
                continue
            sid = int(node.subtask_id)
            prev = best_by_subtask.get(sid)
            if prev is None:
                best_by_subtask[sid] = node
                continue
            rank_cur = (
                node.reward,
                node.get_reward_value(),
                node.visits,
                -node.node_id,
            )
            rank_prev = (
                prev.reward,
                prev.get_reward_value(),
                prev.visits,
                -prev.node_id,
            )
            if rank_cur > rank_prev:
                best_by_subtask[sid] = node

        completed: List[Dict[str, Any]] = []
        for sid in sorted(best_by_subtask.keys()):
            node = best_by_subtask[sid]
            completed.append(
                {
                    "subtask_id": node.subtask_id,
                    "description": node.subtask_description,
                    "result": node.result,
                    "reward": node.reward,
                    "log_path": node.log_path,
                }
            )
        return completed

    # Visualization export
    def serialize_nodes_for_visualization(self) -> List[Dict[str, Any]]:
        payload: List[Dict[str, Any]] = []
        for node in self.tree.get_all_nodes():
            payload.append(
                {
                    "node_id": node.node_id,
                    "parent_id": node.parent.node_id if node.parent else None,
                    "children": [c.node_id for c in node.children],
                    "depth": node.get_depth(),
                    "node_type": node.node_type,
                    "subtask": node.subtask_payload
                    if node.subtask_payload is not None
                    else {
                        "id": node.subtask_id if node.subtask_id > 0 else None,
                        "description": node.subtask_description,
                    },
                    "description": node.subtask_description,
                    "reward": node.reward,
                    "visits": node.visits,
                    "status": node.status,
                    "created_by": node.created_by,
                    "memory": node.memory,
                    "theoretician_output": node.theoretician_output,
                    "supervisor_dispatch": node.supervisor_dispatch or {},
                    "critic_feedback": node.evaluation or {},
                    "supervisor_feedback": node.supervisor_feedback or {},
                    "selected_round": node.selected_round,
                }
            )
        return payload

    # Helpers
    def _get_safe_name(self) -> str:
        instr_name = (
            self.structured_problem.get("instruction_filename")
            or self.structured_problem.get("topic")
        )
        try:
            instr_stem = Path(str(instr_name)).stem
        except Exception:
            instr_stem = str(instr_name)
        return "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(instr_stem))

    def _get_expansion_count_by_node_type(self, node_type: str) -> int:
        ntype = (node_type or "").lower()
        if ntype == "draft":
            return self.draft_expansion
        if ntype == "revise":
            return self.revise_expansion
        return self.revise_expansion

    def _decision_to_node_type(self, decision: str) -> str:
        d = (decision or "").lower()
        if d in ("to_redraft", "complete"):
            return "draft"
        if d == "to_revise":
            return "revise"
        return "revise"

    def _sanitize_node_type(self, node_type: Any, fallback: str) -> str:
        ntype = str(node_type or fallback or "draft").strip().lower()
        if ntype not in {"draft", "revise"}:
            return fallback if fallback in {"draft", "revise"} else "revise"
        return ntype

    def _extract_reward(self, payload: Dict[str, Any]) -> float:
        if not isinstance(payload, dict):
            return 0.0
        value = payload.get("reward")
        try:
            return float(value) if value is not None else 0.0
        except Exception:
            return 0.0

    def _compose_parent_memory(self, parent: Optional[MCTSNode]) -> str:
        if parent is None:
            return ""
        path_nodes = self._get_path_nodes(parent)
        summaries: List[str] = []
        for node in path_nodes:
            summary = self._to_natural_text((node.evaluation or {}).get("summary"))
            if summary:
                summaries.append(summary)
        return "\n".join(summaries)

    def _compose_node_memory(self, node: MCTSNode, critic_feedback: Dict[str, Any]) -> str:
        summary = self._to_natural_text(
            critic_feedback.get("summary")
            or critic_feedback.get("opinion")
            or critic_feedback.get("analysis")
            or ""
        )
        return summary

    def _subtask_brief(self, subtask: Any) -> str:
        if isinstance(subtask, dict):
            sid = subtask.get("id")
            stype = subtask.get("subtask_type")
            desc = str(subtask.get("description") or "").strip()
            bits = []
            if sid is not None:
                bits.append(f"#{sid}")
            if stype:
                bits.append(str(stype))
            if desc:
                bits.append(desc)
            return " | ".join(bits)
        return str(subtask or "").strip()

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
