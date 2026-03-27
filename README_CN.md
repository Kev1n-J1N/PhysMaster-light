# PhysMaster-Light

[English](README.md)

一个轻量级的 AI 物理求解 Agent Pipeline。PhysMaster 将复杂物理问题拆分为子任务，通过 LLM 驱动的推理迭代求解，并生成结构化的解答和可视化结果。

---

## 目录

- [快速开始](#快速开始)
- [基础使用](#基础使用)
- [进阶功能](#进阶功能)
- [可视化](#可视化)

---

## 快速开始

### 1. 克隆与环境配置

```bash
git clone https://github.com/Kev1n-J1N/PhysMaster-light.git
cd PhysMaster-light

# 创建 conda 环境
conda create -n phys python=3.10 -y
conda activate phys

# 安装依赖
pip install -r requirements.txt
```

中国大陆用户可使用镜像源加速安装：

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置 LLM API

编辑 `config.yaml`，设置你的 LLM API：

```yaml
llm:
  base_url: "https://api.openai.com/v1"   # OpenAI 兼容 API 地址
  api_key: "sk-your-api-key-here"          # API 密钥
  model: "gpt-4o"                          # 模型名称
```

PhysMaster 使用 OpenAI 兼容 API 格式，任何支持该格式的提供商均可使用（OpenAI、DeepSeek、本地 vLLM 等）。

### 3. 运行

```bash
conda activate phys
python run.py
```

---

## 基础使用

在基础模式下，PhysMaster 以简单的 **问题澄清 → 求解 → 评估 → 总结** 流水线运行，无需启用任何进阶功能。推荐新手从此模式开始。

### 最小配置

使用基础模式时，在 `config.yaml` 中关闭所有进阶功能：

```yaml
skills:
  enabled: false                         # 关闭技能系统

landau:
  library_enabled: false                 # 关闭 MCP 文献检索
  workflow_enabled: false                # 关闭工作流模板

visualization:
  enabled: true                          # 启用 HTML 可视化
```

### 准备问题文件

创建一个文本文件，写入你的物理问题。支持 LaTeX 数学公式。

**示例** (`instructions/test.txt`)：

```markdown
## 物理问题 — 梯度折射率介质中的光线动力学

一个平面光学介质的折射率随横向坐标 (y) 变化，其关系式为：

[
n(y) = n_0 \left(1 - \frac{1}{2}\alpha y^2 \right), \qquad \alpha > 0
]

其中 (n_0) 和 (α) 为常数。一束单色光沿 (x) 方向传播通过该介质。

利用几何光学原理，将光线轨迹 (y(x)) 视为使光程长度达到极值的路径。

**问题**

推导描述光线轨迹 (y(x)) 的微分方程，并证明在小角度（近轴）近似下，该光线满足简谐振子方程。根据这一结果，用 ( \alpha ) 表示光线在介质内的空间振荡周期。

```

在 `config.yaml` 中设置问题文件路径：

```yaml
pipeline:
  query_file: "instructions/your_problem.txt"
```

### 流水线参数

| 参数 | 说明 | 默认值 |
|---|---|---|
| `pipeline.query_file` | 物理问题文件路径 | `instructions/test.txt` |
| `pipeline.output_path` | 输出目录 | `outputs` |
| `pipeline.parallel_processes` | 求解器并行进程数 | `3` |
| `pipeline.max_rounds` | 所有子任务的最大迭代轮数 | `10` |
| `clarifier.max_key_concpets` | 澄清器提取的最大关键概念数 | `5` |

### 流水线流程

```
问题文件                                                输出
    │                                                   │
    ▼                                                   ▼
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌──────────────┐    ┌────────────┐
│ Clarifier │───▶│ Supervisor │───▶│ Theoretician │───▶│   Critic     │───▶│ Summarizer │
│  澄清器   │    │  调度器     │    │   求解器      │    │   评估器     │    │   总结器    │
└──────────┘    └────────────┘    └──────────────┘    └──────────────┘    └────────────┘
                                          │                    │
                                          └──── 循环 ──────────┘
                                        (修订 / 重做，直到完成或达到 max_rounds)
```

**流程说明：**

1. **Clarifier（澄清器）** — 解析问题，提取关键概念，分解为子任务列表
2. **Supervisor（调度器）** — 为当前子任务生成详细的执行指令，传递给 Theoretician
3. **Theoretician（求解器）** — 执行求解，生成详细解答（可调用 Python、技能、文献检索等工具）
4. **Critic（评估器）** — 评估解答质量，做出决策：
   - `complete` — 当前子任务完成，进入下一个子任务
   - `to_revise` — 方法正确但需改进，Theoretician 修订当前解答
   - `to_redraft` — 方法有误，Theoretician 重新求解
5. **循环** — Supervisor → Theoretician → Critic 持续迭代，直到所有子任务完成或达到 `max_rounds` 上限
6. **Summarizer（总结器）** — 汇总所有已完成子任务，生成最终 Markdown 报告

**关键参数：**
- `max_rounds`：所有子任务的最大迭代轮数（默认 10）。每次 Theoretician 求解计为 1 轮，防止无限循环

### 输出结构

运行后，结果保存在 `outputs/<task_name>/` 目录下：

```
outputs/test/
├── contract.json          # 结构化问题分解
├── node_1/                # 节点 1 的求解输出
├── node_2/                # 节点 2 的求解输出
├── ...
├── summary.md             # 最终总结
└── visualization.html     # 交互式可视化
```

---

## 进阶功能

### 技能系统

技能系统为 Agent 提供特定领域的知识和求解流程。启用后，求解器（Theoretician）会根据问题内容自动匹配并加载相关技能，以增强推理能力。

#### 启用技能

```yaml
skills:
  enabled: true
  roots:
    - "LANDAU/skills"        # 技能目录
```

#### 内置技能

| 技能 | 说明 |
|---|---|
| `classical_electrodynamics` | 麦克斯韦方程、辐射、波导 |
| `quantum_mechanics` | 薛定谔方程、散射、角动量 |
| `thermodynamics_statistical_mechanics` | 配分函数、相变、系综 |
| `conservation_laws` | 诺特定理、守恒流 |
| `perturbation_expansion` | 正则/奇异微扰、渐近级数 |
| `variational_methods` | 欧拉-拉格朗日、瑞利-里兹、变分法 |
| `dimensional_analysis` | Pi 定理、自然单位、标度律 |
| `symmetry_analysis` | 群论、李代数、表示论 |
| `fourier_spectral_analysis` | 傅里叶/拉普拉斯变换、谱方法 |
| `numerical_ode_pde` | 龙格-库塔、有限差分/有限元方法 |
| `statistical_error_analysis` | 误差传播、拟合、蒙特卡洛 |

#### 创建自定义技能

在 `LANDAU/skills/` 下创建新目录，并编写 `SKILL.md` 文件：

```
LANDAU/skills/your_skill_name/
└── SKILL.md
```

**SKILL.md 格式：**
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

技能会在检测到相关问题时自动发现并加载。YAML frontmatter（`name` 和 `description`）用于匹配，完整的 Markdown 内容按需加载。

---

### MCP 文献检索（网页搜索与解析）

文献检索模块通过 MCP（Model Context Protocol）服务器提供网页搜索和内容解析功能。启用后，Supervisor 和 Critic 可以搜索网络参考资料并解析网页获取相关信息。

> MCP 服务器的详细部署与配置说明请参阅 **[mcp_sandbox/README.md](mcp_sandbox/README.md)**。

#### 启用文献检索

在 PhysMaster 的 `config.yaml` 中启用：

```yaml
landau:
  library_enabled: true
  library: "LANDAU/library"

  library_config:
    mcp_url: "http://127.0.0.1:8002/mcp"      # MCP 服务器地址
    search_region: "us"                         # 搜索区域
    search_lang: "en"                           # 搜索语言
    parse_model: "DeepSeek/DeepSeek-V3-0324"    # 网页解析使用的模型
```

#### MCP 工具

| 工具 | 说明 |
|---|---|
| `web_search` | 搜索相关参考资料 |
| `web_parse` | 解析网页并提取相关内容 |

文献检索器通过 Streamable HTTP 传输协议与 MCP 服务器通信。每次工具调用会创建一个临时 MCP 会话，执行工具调用并返回解析结果。

---

### 工作流模板

工作流模板为特定类型的物理问题提供预定义的求解方法论。澄清器使用这些模板生成更结构化的子任务分解。

#### 启用工作流

```yaml
landau:
  workflow_enabled: true
  workflow: "LANDAU/workflow"
```

工作流模板是存储在 `LANDAU/workflow/` 目录中的 YAML 文件，每个模板定义了一类问题的结构化求解方法。workflow 可以定义个性化的问题求解方案，系统主要通过文件名中的关键词匹配来判断是否适用于当前任务，因此构建工作流时需要明确任务类型，使用包含相关关键词的文件名。

---

## 可视化

当 `visualization.enabled` 设为 `true` 时，PhysMaster 会生成求解过程的交互式 HTML 可视化。

```yaml
visualization:
  enabled: true
```

可视化文件（`outputs/<task>/visualization.html`）展示：

- **流水线链** — 求解轨迹中的每个节点，以线性链连接
- **节点详情** — 点击任意节点查看求解输出、评估结果和评分
- **子任务进度** — 每个节点所属子任务及其类型（初稿、修订、重做）
- **总结** — 最终总结 Markdown 嵌入在可视化中

建议下载到本地后直接打开文件进行查看。

---

## 许可

MIT
