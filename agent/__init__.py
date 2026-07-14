"""LLM Eval agent package."""


def __getattr__(name: str):
    if name == "AgentOrchestrator":
        from agent.orchestrator import AgentOrchestrator
        return AgentOrchestrator
    if name == "AgentResult":
        from agent.state import AgentResult
        return AgentResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AgentOrchestrator", "AgentResult"]
