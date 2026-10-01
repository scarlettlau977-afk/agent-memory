# Agent Memory Evaluation Framework

这个子项目使用同一个最小 Agent，对比 No Memory 和 Full History 两种 memory 模式。

## 运行

在仓库根目录执行：

    $env:PYTHONPATH = "$(Get-Location);$(Join-Path (Get-Location) 'src');$(Join-Path (Get-Location) 'agent-memory-eval')"
    python agent-memory-eval\main.py --mode no_memory
    python agent-memory-eval\main.py --mode full_history
    python agent-memory-eval\experiments\baseline.py

默认使用本地 EchoLLM，不需要 API Key。要接入 OpenAI-compatible provider，安装 openai，再通过 OPENAI_API_KEY、OPENAI_MODEL 和可选的 OPENAI_BASE_URL 环境变量提供配置，最后在代码中传入 OpenAICompatibleLLM()。不要提交 .env 或任何真实凭据。

后续的长期记忆、短期记忆、向量检索和评估指标都应实现为新的 MemoryBackend 或独立 adapter，而不改变 MinimalAgent 的调用接口。
