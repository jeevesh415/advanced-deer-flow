import os
import importlib.util
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from recursive_ai.memory.long_term import CognitiveMemory

class MetaArchitect:
    """Agent capable of rewriting the system's own graph definition."""
    def __init__(self, memory: CognitiveMemory, model_name: str = "gpt-4o"):
        self.memory = memory
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    def propose_new_architecture(self, current_graph_path: str, goal: str) -> str:
        """Reads the current graph and proposes a V2."""
        with open(current_graph_path, 'r') as f:
            current_code = f.read()

        prompt = ChatPromptTemplate.from_template(
            """You are a Meta-Architect. The system wants to achieve: {goal}.
            Current Graph Code:
            ```python
            {code}
            ```

            Write a NEW version of this file that optimizes the workflow for the goal.
            For example, add parallel nodes, new specialized agents, or different routing logic.
            Return ONLY the full python code.
            """
        )
        chain = prompt | self.llm
        new_code = chain.invoke({"goal": goal, "code": current_code}).content

        # Clean
        if "```python" in new_code:
            new_code = new_code.split("```python")[1].split("```")[0]
        elif "```" in new_code:
            new_code = new_code.split("```")[1].split("```")[0]

        return new_code

def load_dynamic_graph():
    """Attempts to load 'recursive_ai.graph_v2', falls back to 'recursive_ai.graph'."""
    v2_path = "recursive_ai/graph_v2.py"
    if os.path.exists(v2_path):
        try:
            spec = importlib.util.spec_from_file_location("recursive_ai.graph_v2", v2_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "create_graph"):
                print("🔄 Loaded Dynamic Graph V2")
                return module.create_graph
        except Exception as e:
            print(f"⚠️ Failed to load Graph V2: {e}")

    from recursive_ai.graph import create_graph
    return create_graph
