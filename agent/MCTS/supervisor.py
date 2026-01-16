import multiprocessing as mp
import json
from concurrent.futures import ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .mcts_tree import MCTSTree
from .mcts_node import MCTSNode
from .theoretician import run_theo_node

from utils.gpt5_utils import call_model
from utils.save_utils import MarkdownWriter
from LANDAU.prior.prior_retrive import PriorRetriever

_GLOBAL_POOL: ProcessPoolExecutor | None = None

PRIOR_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "prior_search",
        "description": "Search LANDAU prior knowledge base for relevant chunks.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query."},
                "top_k": {
                    "type": "integer",
                    "description": "Number of chunks to return.",
                    "default": 3,
                },
                "expand_context": {
                    "type": "boolean",
                    "description": "Include prev/next chunks for context.",
                    "default": False,
                },
                "return_format": {
                    "type": "string",
                    "description": "Return text for prompt or raw JSON.",
                    "enum": ["text", "json"],
                    "default": "text",
                },
            },
            "required": ["query"],
        },
    },
}

def _init_worker():
    """Worker initializer: 每个子进程只初始化一次"""
    global _WORKER_INIT
    if "_WORKER_INIT" in globals():
        return
    _WORKER_INIT = True

class SupervisorOrchestrator:
    """MCTS Supervisor，负责完整的树搜索流程"""

    def __init__(
        self,
        structured_problem,
        task_dir: str,
        processes: int = 2,
        max_nodes: int = 4,
        prompts_path: str = "prompts/",
        draft_expansion: int = 2,
        revise_expansion: int = 2,
        improve_expansion: int = 1,
        exploration_constant: float = 1.414,
        beam_width: int = 3,
        landau_prior_enabled: bool = True
    ):
        self.structured_problem = structured_problem
        self.task_dir = task_dir
        self.processes = max(1, processes)
        self.max_nodes = max_nodes
        self.prompts_path = Path(prompts_path)
        self.draft_expansion = draft_expansion
        self.revise_expansion = revise_expansion
        self.improve_expansion = improve_expansion
        self.exploration_constant = exploration_constant
        self.beam_width = max_nodes if beam_width is None else max(1, int(beam_width))
        self.landau_prior_enabled = bool(landau_prior_enabled)
        self._prior_retriever: Optional[PriorRetriever] = None
        self.kb_search_tools: List[Dict[str, Any]] = []
        if self.landau_prior_enabled:
            self.kb_search_tools.append(PRIOR_SEARCH_TOOL)

        prompt_files = {
            "critic_prompt": "critic_prompt.txt",
            "critic_system_prompt": "critic_system_prompt.txt",
            "scheduler_prompt": "scheduler_prompt.txt",
            "scheduler_system_prompt": "scheduler_system_prompt.txt",
            "theoretician_prompt": "theoretician_prompt.txt",
            "theoretician_system_prompt": "theoretician_system_prompt.txt",
        }

        for attr, filename in prompt_files.items():
            setattr(self, attr, self._load_prompt(filename))

        # Build subtasks and initialize tree/root
        self.subtasks = self._build_subtasks()

        self.tree = MCTSTree(
            root_subtask_id=0,
            root_description="Virtual Root",
        )
        self.tree.root.node_type = "virtual"
        self.tree.root.subtask_description = "Virtual Root"
        self.tree.root.subtask_type = "virtual"
        self.tree.root.status = "completed"
        self.tree.root.evaluation = {"decision": "complete", "score": 0}
        self.tree.root.total_reward = 0
        self.tree.root.average_reward = 0
        self.tree.root.visits = 1

        self.node_counter = 1

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

        total_nodes_created = 1 
        completed_subtasks: List[Dict[str, Any]] = []

        while total_nodes_created < self.max_nodes:
            selected_node = self._select_leaf_node()
            if selected_node is None:
                break
            
            scheduler_output: str = None
            try:
                scheduler_output = self._call_scheduler(selected_node) # [subtask_description, background_knowledge]
            except Exception:
                scheduler_output = ""
                print("Scheduler call failed.")

            evaluation = getattr(selected_node, "evaluation", "")
            decision = (evaluation.get("decision") or "to_revise").lower()
            score = float(evaluation.get("score", 0.0) or 0.0)

            expansion_count = self._get_expansion_count(decision)
            child_node_type = self._decision_to_node_type(decision)
            next_subtask_id = selected_node.subtask_id
            is_complete = decision == "complete"

            if selected_node.node_type == "virtual":
                next_subtask_id = self.subtasks[0]["id"] if self.subtasks else 1
                child_node_type = "draft"
                expansion_count = self.draft_expansion

            elif is_complete:
                completed_subtasks.append(
                    {
                        "subtask_id": selected_node.subtask_id,
                        "description": selected_node.subtask_description,
                        "result": getattr(selected_node, "result", None),
                        "score": score,
                        "log_path": getattr(selected_node, "log_path", None),
                        "best_node_index": selected_node.node_index,
                    }
                )
                next_subtask = self._get_next_subtask(selected_node.subtask_id)
                if next_subtask:
                    next_subtask_id = next_subtask["id"]
                    child_node_type = "draft"
                    expansion_count = self.draft_expansion
                else:
                    break
                if len({s["subtask_id"] for s in completed_subtasks}) >= len(self.subtasks):
                    break

            elif decision == "to_redraft":
                child_node_type = "draft"
            elif decision == "to_revise":
                child_node_type = "revise"
            elif decision == "to_improve":
                child_node_type = "improve"
            else:
                child_node_type = "revise"

            new_nodes = self._expand_and_simulate_nodes(
                parent=selected_node,
                node_type=child_node_type,
                count=expansion_count,
                subtask_id=next_subtask_id,
                augmented_description=scheduler_output,
            )
            total_nodes_created += len(new_nodes)

            if total_nodes_created >= self.max_nodes:
                break

        best_trajectory = self._find_best_trajectory()

        summary = {
            "completed_subtasks": completed_subtasks,
            "total_nodes": total_nodes_created,
            "tree_stats": self.tree.get_tree_stats(),
            "best_trajectory": best_trajectory,
        }

        return summary


    def _call_scheduler(self, node: MCTSNode) -> str:
        """调用 Supervisor-Scheduler，根据 structured_problem 给出subtasks及补充信息，并返回调度结果。"""
        if not self.scheduler_prompt:
            return None

        try:
            if node is not None:
                node_info = {
                    "node_index": node.node_index,
                    "subtask_id": node.subtask_id,
                    "node_type": node.node_type,
                    "subtask_description": getattr(node, "subtask_description", None),
                    "evaluation": getattr(node, "evaluation", None),
                    "result": getattr(node, "result", None),
                }
            else:
                node_info = None
        except Exception:
            node_info = None

        system_prompt = self.scheduler_system_prompt
        prompt = self.scheduler_prompt.format(
            structured=json.dumps(self.structured_problem, ensure_ascii=False, indent=2),
            node=json.dumps(node_info, ensure_ascii=False, indent=2),
        )

        tools = self.kb_search_tools
        tool_functions = self._kb_tool_functions()

        response = call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            tools=tools,
            tool_functions=tool_functions,
            model_name="gpt-5",
            markdown_writer=None,
            agent_label="Scheduler",
        )

        return response

    def _call_critic(self, node: MCTSNode) -> Dict[str, Any]:
        """
        Supervisor-Critic 评价节点结果
        Output: decision, score, summary, code
        """
        import json

        result_data = node.result or ""

        if isinstance(result_data, str):
            try:
                node_output = json.loads(result_data)
            except Exception:
                node_output = {}
        else:
            node_output = result_data or {}

        core_results = ""
        analysis = ""
        code = ""
        files: list[Any] = []

        if isinstance(node_output, dict):
            core_results = (
                node_output.get("core_results")
                or node_output.get("core_result")
                or ""
            )
            analysis = node_output.get("analysis") or ""
            code = node_output.get("code") or ""
            files = node_output.get("files") or []

        context_str = json.dumps(
            {
                "analysis": analysis,
                "code": code,
                "files": files,
            },
            ensure_ascii=False,
            indent=2,
        )

        system_prompt = self.critic_system_prompt
        prompt = self.critic_prompt.format(
            result=core_results,
            context=context_str,
        )

        markdown_writer = MarkdownWriter(
            problem=self.structured_problem.get("task_description", ""),
            topic=self.structured_problem.get("topic", ""),
            log_dir=Path(self.task_dir),
            depth=node.get_depth() if node else 0,
            node_index=node.node_index if node else 0,
            file_prefix=self._get_safe_name(),
        )

        tools = self.kb_search_tools
        tool_functions = self._kb_tool_functions()

        response = call_model(
            system_prompt=system_prompt,
            user_prompt=prompt,
            tools=tools,
            tool_functions=tool_functions,
            markdown_writer=markdown_writer,
            agent_label="Critic",
        )

        markdown_writer.write_to_markdown(
            response + "\n",
            mode="critic",
        )

        print("========== Supervisor-Critic ========== \n" + response + "\n")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            return {
                "decision": "to_revise",
                "score": 0.05,
                "summary": "Critic failed to parse model output.",
                "opinion": "Invalid JSON from critic.",
                "code": code,
            }

        decision = str(parsed.get("decision", "to_revise")).strip().lower()
        score = parsed.get("score", 0.0)
        summary = parsed.get("summary") or ""
        opinion = parsed.get("opinion") or ""

        return {
            "decision": decision or "to_revise",
            "score": float(score) if score is not None else 0.0,
            "summary": summary,
            "opinion": opinion,
            "code": code,
        }

    

    def _expand_and_simulate_nodes(
        self,
        parent: MCTSNode,
        node_type: str,
        count: int,
        subtask_id: int,
        augmented_description: str = None,
        ) -> List[MCTSNode]:
        """
        扩展 & 并行模拟多个子节点：
        - 使用Scheduler 由structured_problem生成自然语言任务subtask_description
        - 批量调用 Critic 进行评分，并进行Backpropagation
        """
        if parent and parent.status == "pruned":
            return []

        new_nodes: List[MCTSNode] = []

        subtask_info = next(subtusk for subtusk in self.subtasks if subtusk["id"] == subtask_id)

        futures = []
        indices: List[int] = []
        global _GLOBAL_POOL

        for _ in range(count):
            node_index = self.node_counter
            self.node_counter += 1
            indices.append(node_index)

            parent_memory = ""
            if parent and getattr(parent, "evaluation", None):
                pe = parent.evaluation or {}
                parts = []
                if pe.get("summary"):
                    parts.append(pe["summary"])
                if pe.get("opinion"):
                    parts.append("opinion: " + str(pe["opinion"]))
                if pe.get("core_results"):
                    parts.append("Core results: " + str(pe["core_results"]))
                if pe.get("code"):
                    parts.append("Code: " + str(pe["code"]))
                parent_memory = "\n".join(parts)

            child_depth = (parent.get_depth() + 1) if parent else 0

            payload = {
                "depth": child_depth,
                "node_index": node_index,
                "node_type": node_type,
                "structured_problem": self.structured_problem,
                "subtask": {
                    "id": subtask_id,
                    "description": augmented_description,
                    "subtask_type": subtask_info.get("subtask_type", "reasoning"),
                    "input": subtask_info.get("input"),
                    "expected_output": subtask_info.get("expected_output"),
                },
                "task_dir": self.task_dir,
                "parent_memory": parent_memory,
            }

            futures.append(_GLOBAL_POOL.submit(run_theo_node, payload))

        wait(futures)

        outputs: List[Tuple[MCTSNode, Dict[str, Any]]] = []

        for idx, f in enumerate(futures):
            node_index = indices[idx]

            try:
                node_output = f.result()
            except Exception as e:
                failed_node = MCTSNode(
                    subtask_id=subtask_id,
                    node_index=node_index,
                    node_type=node_type,
                    subtask_description=augmented_description,
                    status="failed",
                )
                failed_node.result = {"error": str(e)}
                failed_node.evaluation = {"decision": "to_revise", "score": 0.0}
                self.tree.add_node(failed_node)
                parent.add_child(failed_node)
                try:
                    failed_node.backpropagate(0.0)
                except Exception:
                    failed_node.update_stats(0.0)
                new_nodes.append(failed_node)
                continue

            child_node = MCTSNode(
                subtask_id=subtask_id,
                node_index=node_index,
                node_type=node_type,
                subtask_description=augmented_description,
                status="completed",
            )
            child_node.result = node_output.get("result")
            child_node.log_path = node_output.get("log_path")
            child_node.subtask_type = subtask_info.get("subtask_type", "reasoning")
            child_node.input = subtask_info.get("input")
            child_node.expected_output = subtask_info.get("expected_output")

            self.tree.add_node(child_node)
            parent.add_child(child_node)

            new_nodes.append(child_node)
            outputs.append((child_node, node_output))

        for child_node, node_output in outputs:
            evaluation = self._call_critic(child_node)
            reward = evaluation.get("score", evaluation.get("reward", 0.0)) or 0.0
            child_node.evaluation = evaluation
            try:
                child_node.backpropagate(reward)
            except Exception:
                child_node.update_stats(reward)

        if new_nodes:
            parent_depth = parent.get_depth() if parent else 0
            self._apply_beam_pruning(parent_depth + 1)

        return new_nodes

    # Tools
    def _get_prior_retriever(self) -> PriorRetriever:
        if self._prior_retriever is None:
            self._prior_retriever = PriorRetriever()
        return self._prior_retriever

    def _prior_search(
        self,
        query: str,
        top_k: int = 3,
        expand_context: bool = False,
        return_format: str = "text",
    ):
        try:
            retriever = self._get_prior_retriever()
            results = retriever.retrieve(
                query=query,
                top_k=int(top_k) if top_k is not None else 3,
                expand_context=bool(expand_context),
            )
            if return_format == "json":
                return results
            return retriever.format_for_llm(results)
        except Exception as e:
            return f"[prior_search] failed: {e}"

    def _kb_tool_functions(self) -> Dict[str, Any]:
        return {"prior_search": self._prior_search} if self.landau_prior_enabled else {}

    def _get_safe_name(self) -> str:
        instr_name = (
            self.structured_problem.get("instruction_filename")
            or self.structured_problem.get("topic")
        )
        try:
            instr_stem = Path(str(instr_name)).stem
        except Exception:
            instr_stem = str(instr_name)
        safe_name = "".join(
            c if (c.isalnum() or c in "._-") else "_" for c in str(instr_stem)
        )
        return safe_name
    
    def _get_expansion_count(self, decision: str) -> int:
        d = (decision or "").lower()
        if d in ("to_redraft", "complete"):
            return self.draft_expansion
        elif d == "to_revise":
            return self.revise_expansion
        elif d == "to_improve":
            return self.improve_expansion
        else:
            return 1

    def _decision_to_node_type(self, decision: str) -> str:
        d = (decision or "").lower()
        if d in ("to_redraft", "complete"):
            return "draft"
        if d == "to_revise":
            return "revise"
        if d == "to_improve":
            return "improve"
        return "revise"

    def _select_leaf_node(self) -> Optional[MCTSNode]:
        """使用 UCB1 策略从树中选择一个要扩展的节点。"""
        
        selected = self.tree.selection(self.exploration_constant)

        if selected is None or selected.status == "pruned":
            return None

        while selected.status == "completed" and selected.children:
            best_child = selected.select_best_child(self.exploration_constant)
            if best_child is None:
                break
            selected = best_child

        return selected

    def _apply_beam_pruning(self, depth: int):
        if self.beam_width is None or self.beam_width <= 0:
            return

        candidates = [
            n for n in self.tree.get_all_nodes()
            if n.status != "pruned" and n.get_depth() == depth
        ]
        if len(candidates) <= self.beam_width:
            return

        def node_reward(node: MCTSNode) -> float:
            if node.evaluation and node.evaluation.get("score") is not None:
                try:
                    return float(node.evaluation.get("score"))
                except Exception:
                    return 0.0
            return float(node.average_reward or 0.0)

        ranked = sorted(
            candidates,
            key=lambda n: (node_reward(n), -n.node_index),
            reverse=True,
        )
        keep = {n.node_index for n in ranked[: self.beam_width]}
        for node in candidates:
            if node.node_index not in keep:
                node.status = "pruned"

    def _build_subtasks(self) -> List[Dict[str, Any]]:
        """ 直接读取 structured_problem 中已有的 sub-tasks """
        subtasks_payload = self.structured_problem.get("sub-tasks", [])

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

        # Ensure deterministic ordering by id
        try:
            subtasks_payload = sorted(subtasks_payload, key=lambda x: x.get("id", 0))
        except Exception:
            pass
        return subtasks_payload

    def _get_next_subtask(self, current_subtask_id: int) -> Optional[Dict[str, Any]]:
        """获取下一个 subtask"""
        for subtask in self.subtasks:
            if subtask["id"] > current_subtask_id:
                return subtask
        return None

    def _find_best_trajectory(self) -> List[Dict[str, Any]]:
        """
        找到覆盖所有 subtasks 的一条最优路径，从虚拟根往下
        """
        best_path: List[MCTSNode] = []
        best_avg_reward = -float("inf")
        best_partial_path: List[MCTSNode] = []
        best_partial_score = -float("inf")

        def dfs(node: MCTSNode, current_path: List[MCTSNode], current_subtask_id: int):
            nonlocal best_path, best_avg_reward, best_partial_path, best_partial_score

            if node.node_type != "virtual":
                current_path.append(node)

            rewards = [n.average_reward for n in current_path if n.visits > 0]
            avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
            node_score = avg_reward
            if node.evaluation and node.evaluation.get("score") is not None:
                try:
                    node_score = float(node.evaluation.get("score"))
                except Exception:
                    node_score = avg_reward

            if node.evaluation and node.evaluation.get("decision") == "complete":
                if avg_reward > best_avg_reward:
                    best_avg_reward = avg_reward
                    best_path = list(current_path)

            if node_score > best_partial_score:
                best_partial_score = node_score
                best_partial_path = list(current_path)

            for child in node.children:
                if child.subtask_id >= current_subtask_id:
                    dfs(child, current_path, max(current_subtask_id, child.subtask_id))

            if node.node_type != "virtual":
                current_path.pop()

        dfs(self.tree.root, [], self.tree.root.subtask_id)

        trajectory = best_path if best_path else best_partial_path
        return [
            {
                "node_index": n.node_index,
                "subtask_id": n.subtask_id,
                "node_type": n.node_type,
                "description": n.subtask_description,
                "result": getattr(n, "result", None),
                "score": n.evaluation.get("score") if n.evaluation else None,
                "log_path": getattr(n, "log_path", None),
            }
            for n in trajectory
        ]
