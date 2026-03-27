# PhysMaster-Light

[中文文档](README_CN.md)

A lightweight AI agent pipeline for solving physics problems. PhysMaster decomposes complex physics problems into subtasks, solves them iteratively with LLM-powered reasoning, and produces structured solutions with optional visualization.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Basic Usage](#basic-usage)
- [Advanced Features](#advanced-features)
- [Visualization](#visualization)

---

## Quick Start

### 1. Clone & Setup Environment

```bash
git clone https://github.com/Kev1n-J1N/PhysMaster-light.git
cd PhysMaster-light

# Create conda environment
conda create -n phys python=3.10 -y
conda activate phys

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure LLM API

Edit `config.yaml` and set your LLM endpoint:

```yaml
llm:
  base_url: "https://api.openai.com/v1"   # Your OpenAI-compatible API endpoint
  api_key: "sk-your-api-key-here"          # Your API key
  model: "gpt-4o"                          # Model name
```

PhysMaster uses the OpenAI-compatible API format. Any provider that supports this format will work (OpenAI, DeepSeek, local vLLM, etc.).

### 3. Run

```bash
conda activate phys
python run.py
```

---

## Basic Usage

In the basic mode, PhysMaster works as a simple **Clarifier → Solver → Critic → Summarizer** pipeline without any advanced features. This is recommended for getting started.

### Minimal Configuration

To run in basic mode, disable all advanced features in `config.yaml`:

```yaml
skills:
  enabled: false                         # Disable skills

landau:
  library_enabled: false                 # Disable MCP library search
  workflow_enabled: false                # Disable workflow templates

visualization:
  enabled: true                          # Enable HTML visualization
```

### Preparing Your Query

Create a text file with your physics problem. The file can contain LaTeX math notation.

**Example** (`instructions/test.txt`):

```markdown
## Physics Problem — Ray Dynamics in a Gradient-Index Medium

A planar optical medium has a refractive index that varies with the transverse coordinate ( y ) according to

[
n(y) = n_0 \left(1 - \frac{1}{2}\alpha y^2 \right), \qquad \alpha > 0
]

where ( n_0 ) and ( \alpha ) are constants. A monochromatic light ray propagates through this medium in the (x)-direction.

Using the principles of geometrical optics, treat the ray trajectory (y(x)) as the path that extremizes the optical path length.

**Problem**

Derive the differential equation governing the ray trajectory (y(x)) and show that, under the small-angle (paraxial) approximation, the ray satisfies a simple harmonic oscillator equation. From this result, determine the spatial oscillation period of the ray inside the medium in terms of ( \alpha ).

```

Set the query file path in `config.yaml`:

```yaml
pipeline:
  query_file: "instructions/your_problem.txt"
```

### Pipeline Parameters

| Parameter | Description | Default |
|---|---|---|
| `pipeline.query_file` | Path to the physics problem file | `instructions/test.txt` |
| `pipeline.output_path` | Output directory | `outputs` |
| `pipeline.parallel_processes` | Worker processes for theoretician | `3` |
| `pipeline.max_rounds` | Max solver-critic iterations across all subtasks | `10` |
| `clarifier.max_key_concpets` | Max key concepts the clarifier extracts | `5` |

### Pipeline Flow

```
Query File                                                Output
    │                                                       │
    ▼                                                       ▼
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│ Clarifier │───▶│ Supervisor │───▶│ Theoretician │───▶│   Critic     │───▶│ Summarizer │
└──────────┘    └────────────┘    └──────────────┘    └──────────────┘    └────────────┘
                                          │                    │
                                          └──── loop ──────────┘
                                        (revise / redraft until complete or max_rounds)
```

1. **Clarifier** — Parses the problem, extracts key concepts, and decomposes it into a subtask list.
2. **Supervisor** — Generates detailed execution instructions for the current subtask and passes them to the Theoretician.
3. **Theoretician** — Executes the solve, producing a detailed solution (can invoke Python, skills, library search, etc.).
4. **Critic** — Evaluates solution quality and makes a decision:
   - `complete` — Current subtask is done; move to the next subtask.
   - `to_revise` — Approach is correct but needs improvement; Theoretician revises the current solution.
   - `to_redraft` — Approach is flawed; Theoretician re-solves from scratch.
5. **Loop** — Supervisor → Theoretician → Critic iterates until all subtasks are complete or the `max_rounds` limit is reached.
6. **Summarizer** — Aggregates all completed subtasks into a final Markdown report.

**Key parameter:**
- `max_rounds`: Maximum iteration rounds across all subtasks (default 10). Each Theoretician solve counts as 1 round, preventing infinite loops.

### Output Structure

After running, results are saved in `outputs/<task_name>/`:

```
outputs/test/
├── contract.json          # Structured problem decomposition
├── node_1/                # Solver output for node 1
├── node_2/                # Solver output for node 2
├── ...
├── summary.md             # Final summary
└── visualization.html     # Interactive visualization
```

---

## Advanced Features

### Skills System

Skills provide domain-specific knowledge and problem-solving workflows to the agent. When enabled, the solver (Theoretician) automatically matches and loads relevant skills based on the problem content to enhance its reasoning capabilities.

#### Enable Skills

```yaml
skills:
  enabled: true
  roots:
    - "LANDAU/skills"
```

#### Built-in Skills

| Skill | Description |
|---|---|
| `classical_electrodynamics` | Maxwell's equations, radiation, waveguides |
| `quantum_mechanics` | Schrodinger equation, scattering, angular momentum |
| `thermodynamics_statistical_mechanics` | Partition functions, phase transitions, ensembles |
| `conservation_laws` | Noether's theorem, conserved currents |
| `perturbation_expansion` | Regular/singular perturbation, asymptotic series |
| `variational_methods` | Euler-Lagrange, Rayleigh-Ritz, calculus of variations |
| `dimensional_analysis` | Pi theorem, natural units, scaling laws |
| `symmetry_analysis` | Group theory, Lie algebras, representation theory |
| `fourier_spectral_analysis` | Fourier/Laplace transforms, spectral methods |
| `numerical_ode_pde` | Runge-Kutta, finite difference/element methods |
| `statistical_error_analysis` | Error propagation, fitting, Monte Carlo |

#### Create Custom Skills

Create a new directory under `LANDAU/skills/` with a `SKILL.md` file:

```
LANDAU/skills/your_skill_name/
└── SKILL.md
```

**SKILL.md format:**
```markdown
---
name: "your_skill_name"
description: "Brief description of when to use this skill."
---

# Your Skill Name

Apply this skill when the problem involves ...

## Goal
What this skill aims to achieve.

## Scope
- Topic 1
- Topic 2

## Inputs
- `parameter_1`: Description
- `parameter_2`: Description

## Outputs
- `result_1`: Description

## Workflow
1. Step one ...
2. Step two ...

## Quality Checks
- Check 1 ...
- Check 2 ...

## Constraints
- Constraint 1 ...
```

Skills are automatically discovered and loaded when relevant problems are detected. The YAML frontmatter (`name` and `description`) is used for matching; the full markdown body is loaded on demand.

---

### MCP Library (Web Search & Parse)

The library module provides web search and content parsing via an MCP (Model Context Protocol) server. When enabled, the supervisor and critic can search the web for reference materials and parse web pages for relevant information.

> For detailed MCP server deployment and configuration, see **[mcp_sandbox/README.md](mcp_sandbox/README.md)**.

#### Enable Library

In PhysMaster's `config.yaml`:

```yaml
landau:
  library_enabled: true
  library: "LANDAU/library"

  library_config:
    mcp_url: "http://127.0.0.1:8002/mcp"      # MCP server endpoint
    search_region: "us"                         # Search region
    search_lang: "en"                           # Search language
    parse_model: "DeepSeek/DeepSeek-V3-0324"    # Model for web page parsing
```

#### MCP Tools

| Tool | Description |
|---|---|
| `web_search` | Search the web for relevant references |
| `web_parse` | Parse a web page and extract relevant content |

The library retriever communicates with the MCP server using the Streamable HTTP transport protocol. Each tool call opens a short-lived MCP session, calls the tool, and returns the parsed result.

---

### Workflow Templates

Workflow templates provide predefined problem-solving methodologies for specific types of physics problems. The clarifier uses these templates to generate more structured subtask decompositions.

#### Enable Workflow

```yaml
landau:
  workflow_enabled: true
  workflow: "LANDAU/workflow"
```

Workflow templates are YAML files stored in the `LANDAU/workflow/` directory. Each template defines a structured methodology for a class of problems. Workflows allow you to define custom problem-solving strategies. The system matches workflows to tasks primarily through filename keywords, so when creating a workflow, use a filename that shares relevant keywords with the target task type.

---

## Visualization

When `visualization.enabled` is set to `true`, PhysMaster generates an interactive HTML visualization of the solving process.

```yaml
visualization:
  enabled: true
```

The visualization (`outputs/<task>/visualization.html`) shows:

- **Pipeline chain** — each node in the solving trajectory, connected as a linear chain
- **Node details** — click any node to see the solver output, critic evaluation, and reward score
- **Subtask progress** — which subtask each node belongs to and whether it was a draft, revision, or redraft
- **Summary** — the final summary markdown is embedded in the visualization

It is recommended to download the HTML file locally and open it in a browser for viewing.

---

## License

MIT
