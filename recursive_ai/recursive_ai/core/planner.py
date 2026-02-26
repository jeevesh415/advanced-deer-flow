from typing import List, Dict, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import BaseMessage

class Thought(TypedDict):
    step: str
    reasoning: str
    score: float # 0.0 to 1.0

class ToTPlanner:
    """Implements Tree of Thoughts planning."""
    def __init__(self, model_name: str = "gpt-4o"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.7)
        self.evaluator = ChatOpenAI(model=model_name, temperature=0.1)

    def generate_thoughts(self, task: str, history: List[BaseMessage], k: int = 3) -> List[Thought]:
        """Generates k potential next steps."""
        prompt = ChatPromptTemplate.from_template(
            """You are an elite AI planner. Task: {task}
            History: {history}

            Generate {k} distinct, creative next steps to advance the task.
            For each step, explain briefly why it is good.
            Format:
            1. Step: [Action] | Reasoning: [Why]
            2. Step: [Action] | Reasoning: [Why]
            ...
            """
        )
        response = prompt | self.llm
        output = response.invoke({"task": task, "history": str(history[-2:]), "k": k}).content

        thoughts = []
        for line in output.split("\n"):
            if "Step:" in line and "|" in line:
                try:
                    parts = line.split("|")
                    step_part = parts[0].split("Step:")[1].strip()
                    reasoning_part = parts[1].split("Reasoning:")[1].strip()
                    thoughts.append({"step": step_part, "reasoning": reasoning_part, "score": 0.0})
                except:
                    continue
        return thoughts

    def evaluate_thoughts(self, task: str, thoughts: List[Thought]) -> List[Thought]:
        """Scores each thought based on likelihood of success."""
        # This implementation iterates through the list, invoking the LLM for each thought.
        # This could be slow for many thoughts, but is acceptable for k=3.
        updated_thoughts = []
        for thought in thoughts:
            prompt = ChatPromptTemplate.from_template(
                """Task: {task}
                Proposed Step: {step}
                Reasoning: {reasoning}

                Rate this step from 0.0 to 1.0 based on feasibility and alignment with the goal.
                Return ONLY the number.
                """
            )
            chain = prompt | self.evaluator
            try:
                result = chain.invoke({
                    "task": task,
                    "step": thought['step'],
                    "reasoning": thought['reasoning']
                })
                score = float(result.content.strip())
                thought['score'] = score
            except Exception:
                thought['score'] = 0.5 # Default fallback
            updated_thoughts.append(thought)

        return updated_thoughts

    def select_best_step(self, task: str, history: List[BaseMessage]) -> str:
        """Main entry point: Generate -> Evaluate -> Select."""
        thoughts = self.generate_thoughts(task, history)
        thoughts = self.evaluate_thoughts(task, thoughts)

        if not thoughts:
            return "research" # Safe fallback

        best_thought = max(thoughts, key=lambda x: x['score'])

        # Mapping back to graph nodes (simplified)
        action = best_thought['step'].lower()
        if "research" in action or "search" in action or "find" in action:
            return "research"
        elif "code" in action or "implement" in action or "write" in action:
            return "code"
        elif "experiment" in action or "test" in action or "try" in action:
            return "experiment"
        elif "swarm" in action or "parallel" in action or "split" in action:
            return "swarm"
        elif "finish" in action or "complete" in action:
            return "finish"
        else:
            return "research" # Default to gathering more info

def create_tot_planner() -> ToTPlanner:
    return ToTPlanner()
