# Agent Memory Evaluation Framework

这个子项目使用同一个最小 Agent，对比 No Memory 和 Full History 两种 memory 模式。

其中 `NoMemory` 是 **Baseline 0**：它实现统一的 `BaseMemory` 接口，但不会保存任何 interaction，`retrieve()` 永远返回 `[]`。

```text
User -> Agent -> NoMemory.retrieve() == [] -> LLM
```

## 运行

在仓库根目录执行：

    $env:PYTHONPATH = "$(Get-Location);$(Join-Path (Get-Location) 'src');$(Join-Path (Get-Location) 'agent-memory-eval')"
    python agent-memory-eval\main.py --mode no_memory
    python agent-memory-eval\main.py --mode full_history
    python agent-memory-eval\experiments\baseline.py

默认使用本地 EchoLLM，不需要 API Key。要接入 OpenAI-compatible provider，安装 openai，再通过 OPENAI_API_KEY、OPENAI_MODEL 和可选的 OPENAI_BASE_URL 环境变量提供配置，最后在代码中传入 OpenAICompatibleLLM()。不要提交 .env 或任何真实凭据。

后续的长期记忆、短期记忆、向量检索和评估指标都应实现为新的 MemoryBackend 或独立 adapter，而不改变 MinimalAgent 的调用接口。

## 第一个 Memory Experiment

`dataset/conversations.json` 保存 Alice 的四轮编程语言偏好对话。运行 `experiments/baseline.py`，脚本会对比同一 Agent 使用 `NoMemory` 与 `FullHistoryMemory` 时的最终答案。默认使用确定性 `PreferenceProbeLLM` 检查 prompt 是否拿到了历史事实；它是测试替身，不是真实 LLM，也不能用于声称模型性能。接入 OpenAI-compatible LLM 后，可在同一 Agent 与数据上做真实模型实验。

## Evaluation 结果

运行 baseline 后，结果会保存到 `experiments/results/baseline_results.json`。每条记录包含问题、标准答案、检索结果、Agent 答案和正确性：

```json
{
  "question": "What programming language does the user prefer?",
  "ground_truth": "Python",
  "retrieved_memories": [],
  "answer": "I don't know.",
  "correct": false,
  "diagnosis": "memory_failure"
}
```

`diagnosis` 的初版规则是：事实没有存下来时为 `memory_failure`；事实已存在但没有被检索到时为 `retrieval_failure`；事实已被检索但答案仍错误时为 `agent_reasoning_failure`；答案正确时为 `correct`。
