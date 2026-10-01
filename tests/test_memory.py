from agent_memory import AgentMemory, ConstructionCase, MemoryTier, RetrievalCase, evaluate_construction, evaluate_retrieval


def test_construct_routes_to_tiers_and_retrieves_from_both():
    memory = AgentMemory()
    short_decision, short = memory.construct("当前任务要使用 Python", user_id="u", session_id="s")
    long_decision, long = memory.construct("请记住我喜欢喝绿茶", user_id="u", session_id="s")
    assert short_decision.tier == MemoryTier.SHORT_TERM
    assert long_decision.tier == MemoryTier.LONG_TERM
    results = memory.retrieve("绿茶 Python", user_id="u", session_id="s", top_k=5)
    assert {r.memory.id for r in results} == {short.id, long.id}


def test_evaluation_reports_construction_and_retrieval():
    memory = AgentMemory()
    cases = [ConstructionCase("请记住我住在上海", True, MemoryTier.LONG_TERM), ConstructionCase("你好", False)]
    report = evaluate_construction(memory, cases)
    assert report.metrics["f1"] > 0.5
    _, item = memory.construct("我喜欢跑步", user_id="eval-user", session_id="eval-session")
    retrieval = evaluate_retrieval(memory, [RetrievalCase("喜欢跑步", {item.id})])
    assert retrieval.metrics["recall@k"] == 1.0
