# PHY_Master_v2 — MCTS Supervisor

This repository contains an MCTS-based supervisor/orchestrator for solving structured problems.

Summary of important behaviors changed recently:

- Output folders now use the instruction file name stem (no extension) as the top-level folder under `outputs/`.
  - Example: `outputs/my_instruction/depth0_node0/...`

- Supervisor decision taxonomy (the Supervisor Critic must return exactly one of the tokens):
  - `to_revise`: Current solution has errors; produce revised nodes that correct mistakes.
  - `to_improve`: Current solution is broadly correct; produce improved nodes that refine/extend the solution.
  - `to_redraft`: Current approach is fundamentally wrong; generate new drafts (alternative approaches).
  - `complete`: Current subtask is sufficiently solved; proceed to draft next subtask.

- Node types used in MCTS: `draft`, `revise`, `improve`. Child nodes are created with these node types according to the supervisor decision mapping.

- Configuration options (in `config.yaml`):
  - `pipeline.complete_score_threshold` (float): Supervisor confidence threshold for treating a subtask as complete. Default `0.9`.
  - `mcts.improve_expansion` (int): Number of parallel children to generate for an `improve` decision. Default `2`.

- Prompt changes: `prompts/supervisor_prompt.txt` now instructs the Supervisor to return a strict JSON object with `decision` equal to one of the four canonical tokens above.

If you want the `complete_score_threshold` or expansion counts adjusted, edit `config.yaml` under the `pipeline` and `mcts` sections.

