from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage
from recursive_ai.core.protocol import AgentStatus

class Reflector:
    """Agent that reflects on past actions to improve future strategy."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.1)
        self.strategy_file = "recursive_ai/knowledge/strategy.md"

    def reflect_on_execution(self, messages: List[BaseMessage], task: str, success: bool):
        """Analyzes a trace and updates the strategy file."""

        # 1. Summarize the run
        history_text = "\n".join([f"{m.type}: {m.content[:500]}..." for m in messages])

        prompt = f"""You are a Meta-Cognitive Engine. Analyze this execution trace for task: '{task}'.
        Outcome: {'SUCCESS' if success else 'FAILURE'}

        Trace:
        {history_text}

        Identify ONE key lesson or 'Rule of Thumb' to improve future performance.
        Format: "Rule: [Brief Description]"
        If no new lesson, return "None".
        """

        response = self.llm.invoke([SystemMessage(content=prompt)]).content

        if "Rule:" in response:
            self._update_strategy_file(response)
            return response
        return "No new insights."

    def _update_strategy_file(self, new_rule: str):
        """Appends the new rule to the strategy file."""
        import os
        os.makedirs(os.path.dirname(self.strategy_file), exist_ok=True)

        with open(self.strategy_file, "a+") as f:
            f.seek(0)
            content = f.read()
            if new_rule not in content:
                f.write(f"\n- {new_rule}")

def create_reflector() -> Reflector:
    return Reflector()
