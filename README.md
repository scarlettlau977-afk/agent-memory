# Pluggable Agent Memory Framework

一个面向 Agent 的可插拔记忆基线，覆盖记忆构建、分层存储、检索和下游效果评估。项目默认使用纯 Python 的启发式策略与内存存储，便于测试和替换为生产实现。

## 功能

- 通过 MemoryPolicy 判断内容是否值得记忆，并路由到短期或长期记忆。
- 通过 MemoryStore 抽象存储后端，支持按用户、会话、标签和查询检索。
- 提供过期清理、隐私删除、用户隔离及访问统计。
- 提供 construction、retrieval 和 downstream 三类评估指标。
- agent-memory-eval/ 包含最小 Agent，用于直观比较无记忆和完整历史两种模式。
- 第二阶段实验提供确定性偏好问答基线、错误归因和 JSON 结果输出。

## 安装

需要 Python 3.10 或更高版本。推荐在虚拟环境中安装：

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python -m pip install -e .
    python -m pip install pytest

## 快速使用

    from agent_memory import AgentMemory

    memory = AgentMemory()
    memory.construct("请记住我喜欢喝绿茶", user_id="u1", session_id="s1")
    memory.construct("当前任务要使用 Python", user_id="u1", session_id="s1")
    print(memory.context("用户喜欢什么", user_id="u1", session_id="s1"))

实现自定义策略时继承 MemoryPolicy 并实现 decide()；实现自定义存储时继承 MemoryStore 并实现 put、delete、search 和 all，即可接入 Redis、Postgres 或向量数据库。

## 运行测试

    python -m pytest -q

当前测试覆盖核心记忆流程、构建与检索评估、Agent baseline，以及 Recall@K、HitRate@K、MRR 和 nDCG@K 的边界行为。

## 运行评估示例

    $env:PYTHONPATH = "$(Get-Location);$(Join-Path (Get-Location) 'src');$(Join-Path (Get-Location) 'agent-memory-eval')"
    python agent-memory-eval\main.py --mode no_memory
    python agent-memory-eval\main.py --mode full_history
    python agent-memory-eval\experiments\baseline.py

评估数据位于 agent-memory-eval/dataset/conversations.json。该子项目默认使用本地 EchoLLM，不需要 API Key；可选的 OpenAI-compatible 适配器会延迟导入 openai，凭据应通过环境变量提供，切勿提交到仓库。

实验结果默认写入 agent-memory-eval/experiments/results/baseline_results.json。该结果用于验证记忆链路，不代表真实模型性能。

## 评估指标

- evaluate_construction：precision、recall、F1 和记忆层路由准确率。
- evaluate_retrieval：Recall@K、HitRate@K、MRR、nDCG@K，并保留每条 query 的返回 ID。
- evaluate_downstream：比较有记忆与无记忆上下文的平均任务得分增益。

### Retrieval 指标定义

- `Recall@K` 是 Top K 中命中的唯一相关记忆数量除以该查询全部相关记忆数量；例如 4 条相关记忆中命中 1 条时，`Recall@3 = 0.25`。
- `HitRate@K` 只表示 Top K 是否至少命中一条相关记忆；同一例子的 `HitRate@3 = 1.0`。它与 Recall@K 是两个独立指标。
- `MRR` 对每条查询取第一条相关结果排名的倒数，再对查询做算术平均；Top K 内没有命中时该查询为 0。
- `nDCG@K` 使用二元相关性和排名折扣衡量排序质量，理想排序为 1.0。
- 相关集合为空时，单查询 Recall 未定义并记为 `null`，宏平均 Recall 排除该查询并记录 `recall_defined_query_count`；HitRate、MRR 和 nDCG 记为 0.0。
- 结果按已有 memory ID 匹配，重复 ID 只计一次，`top_k` 必须是正整数。

## 项目结构

    src/agent_memory/        核心记忆框架
    tests/                   自动化测试
    agent-memory-eval/       最小 Agent 与基线评估

## 许可证

当前仓库未指定许可证。如需公开复用，请在发布前补充合适的 LICENSE 文件。
