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
cd PHY_Master

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
# Radiation and Self-Energy of a Uniformly Moving Charge

A point charge q moves with constant velocity v = v*z_hat in vacuum.

## 1. Fields
Using Liénard-Wiechert potentials, derive E(r, t) and B(r, t).

## 2. Energy
Compute the Poynting vector and show there is no radiation.

## 3. Relativistic Covariance
Write the field tensor F^{μν} in the rest frame and apply a Lorentz boost.
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
Query File                          Output
    │                                 │
    ▼                                 ▼
┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌────────────┐
│ Clarifier │───▶│  Solver   │───▶│   Critic     │───▶│ Summarizer │
└──────────┘    └───────────┘    └──────────────┘    └────────────┘
                     │                  │
                     └──── loop ────────┘
                   (revise / redraft)
```

1. **Clarifier** parses the problem, extracts key concepts, and decomposes it into subtasks.
2. **Solver** (Theoretician) generates a detailed solution for each subtask.
3. **Critic** evaluates the solution and decides: `complete` | `to_revise` | `to_redraft`.
4. If not complete, the solver revises or redrafts the solution based on critic feedback.
5. **Summarizer** produces a final markdown summary of all completed subtasks.

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

Skills provide domain-specific knowledge and problem-solving workflows to the agent. When enabled, the clarifier and solver automatically load relevant skills based on the problem content.

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
| `web_search` | Search the web for physics references |
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
