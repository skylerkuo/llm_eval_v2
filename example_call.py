from agent.orchestrator import AgentOrchestrator


def main() -> None:

    agent = AgentOrchestrator()

    question = "束線帶是做什麼用的？"
    print(f"question: {question}\n")

    result = agent.run(
        question,
        send_robot=False, 
        record_memory=True,
        session_id="demo",
    )

    print(f"intent: {result.intent}")
    print(f"response: {result.response_text}")

    if result.intent == "plan":
        print(f"steps: {result.payload.get('steps')}")
    elif result.intent == "robot_action":
        print(f"actions: {result.payload.get('actions')}")
    elif result.error:
        print(f"error: {result.error}")


if __name__ == "__main__":
    main()
