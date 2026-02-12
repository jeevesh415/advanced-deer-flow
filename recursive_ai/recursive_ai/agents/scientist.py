import os
import uuid
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from recursive_ai.agents.evolution import SoftwareEngineer
from recursive_ai.memory.long_term import CognitiveMemory

class ScientistAgent:
    """Agent dedicated to abstract experimentation and hypothesis testing."""
    def __init__(self, memory: CognitiveMemory, model_name: str = "gpt-4o"):
        self.memory = memory
        self.engineer = SoftwareEngineer(memory, model_name)
        self.llm = ChatOpenAI(model=model_name, temperature=0.7) # Higher temp for creativity

    def conduct_experiment(self, goal: str) -> str:
        """Formulates a hypothesis and tests it in a sandbox."""

        # 1. Formulate Hypothesis
        hypothesis_prompt = ChatPromptTemplate.from_template(
            """You are a radical AI scientist. The system goal is: {goal}.
            Propose a risky, innovative experiment to improve performance or capability.
            Focus on architectural changes, new algorithms, or optimization.

            Hypothesis:"""
        )
        hypothesis = hypothesis_prompt | self.llm
        hypo_text = hypothesis.invoke({"goal": goal}).content

        # 2. Design Experiment
        experiment_id = str(uuid.uuid4())[:8]
        sandbox_dir = f"recursive_ai/sandbox/{experiment_id}"
        os.makedirs(sandbox_dir, exist_ok=True)

        design_prompt = ChatPromptTemplate.from_template(
            """Design a python script to test this hypothesis: {hypo}.
            The script should be self-contained and print 'SUCCESS' or 'FAILURE' at the end.
            """
        )
        design = design_prompt | self.llm
        script_content = design.invoke({"hypo": hypo_text}).content

        # Clean up code block
        if "```python" in script_content:
            script_content = script_content.split("```python")[1].split("```")[0]
        elif "```" in script_content:
            script_content = script_content.split("```")[1].split("```")[0]

        script_path = f"{sandbox_dir}/experiment.py"
        with open(script_path, "w") as f:
            f.write(script_content)

        # 3. Execute
        # We need to simulate the execution. Since SoftwareEngineer tools return "STDOUT: ...", we parse that.
        result = self.engineer.execute_code_action(f"python {script_path}")

        # 4. Record Findings
        report = f"Experiment {experiment_id}\nHypothesis: {hypo_text}\nResult: {result}"
        self.memory.add_memory(report, metadata={"type": "experiment", "id": experiment_id})

        return report

def create_scientist(memory: CognitiveMemory) -> ScientistAgent:
    return ScientistAgent(memory)
