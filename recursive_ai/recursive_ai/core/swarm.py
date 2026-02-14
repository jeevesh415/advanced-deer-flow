import asyncio
from typing import List, Dict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from recursive_ai.agents.acquisition import ResearchAgent
from recursive_ai.agents.evolution import SoftwareEngineer
from recursive_ai.memory.long_term import CognitiveMemory

class SwarmManager:
    """Manages parallel execution of multiple agents."""
    def __init__(self, memory: CognitiveMemory, model_name: str = "gpt-4o"):
        self.memory = memory
        self.llm = ChatOpenAI(model=model_name, temperature=0)

    async def decompose_and_execute(self, task: str) -> str:
        """Splits task into subtasks and runs them in parallel."""

        # 1. Decompose
        prompt = ChatPromptTemplate.from_template(
            """You are a Swarm Commander. Split this complex task into 3-5 independent subtasks.
            Task: {task}

            Return ONLY a python list of strings.
            Example: ["Research X", "Write module Y", "Test Z"]
            """
        )
        response = self.llm.invoke({"task": task})
        try:
            subtasks = eval(response.content)
            if not isinstance(subtasks, list):
                subtasks = [task]
        except:
            subtasks = [task]

        print(f"🐝 Swarm activated. Subtasks: {subtasks}")

        # 2. Assign & Execute (Simulated Async)
        # In a real environment, we'd use true asyncio.gather with async agent methods.
        # Since our agents currently use sync calls, we wrap them in threads or just loop for this POC.
        # To be "Advanced", let's use asyncio.to_thread
        tasks = []
        for st in subtasks:
             tasks.append(self._execute_subtask(st))

        results = await asyncio.gather(*tasks)

        # 3. Aggregation
        summary = "\n".join([f"Subtask {i+1}: {res}" for i, res in enumerate(results)])
        return f"Swarm Execution Complete.\n{summary}"

    async def _execute_subtask(self, subtask: str) -> str:
        """Routes a subtask to the appropriate agent."""
        # Simple routing heuristic
        if "research" in subtask.lower() or "find" in subtask.lower():
            agent = ResearchAgent(self.memory)
            return await asyncio.to_thread(agent.perform_research, subtask)
        else:
            agent = SoftwareEngineer(self.memory)
            return await asyncio.to_thread(agent.implement_feature, subtask)

def create_swarm_manager(memory: CognitiveMemory) -> SwarmManager:
    return SwarmManager(memory)
