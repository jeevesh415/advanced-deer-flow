import os
import subprocess
from typing import List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from recursive_ai.core.protocol import AgentStatus, Task
from recursive_ai.memory.long_term import CognitiveMemory
from recursive_ai.memory.skills import SkillLibrary
from recursive_ai.core.simulation import create_simulator

# Define tools
@tool
def list_files(path: str = ".") -> str:
    """Lists files in a directory."""
    try:
        return str(os.listdir(path))
    except Exception as e:
        return f"Error: {e}"

@tool
def read_file(filepath: str) -> str:
    """Reads a file's content."""
    try:
        with open(filepath, "r") as f:
            return f.read()
    except Exception as e:
        return f"Error: {e}"

@tool
def write_file(filepath: str, content: str) -> str:
    """Writes content to a file. Overwrites if exists."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w") as f:
            f.write(content)
        return f"File {filepath} written successfully."
    except Exception as e:
        return f"Error: {e}"

@tool
def run_command(command: str) -> str:
    """Runs a shell command and returns output."""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        return f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    except Exception as e:
        return f"Error: {e}"

class SoftwareEngineer:
    """Agent capable of modifying the codebase."""
    def __init__(self, memory: CognitiveMemory, model_name: str = "gpt-4o"):
        self.memory = memory
        self.skills = SkillLibrary()
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tools = [list_files, read_file, write_file, run_command]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.status = AgentStatus.IDLE

    def implement_feature(self, task_description: str) -> str:
        """Writes code to satisfy a requirement."""
        self.status = AgentStatus.WORKING

        # 0. Retrieve Skills
        try:
            relevant_skills = self.skills.retrieve_skill(task_description)
            skills_context = "\n".join([doc.page_content for doc in relevant_skills])
        except Exception:
            skills_context = "No relevant skills found."

        # 1. Plan
        planning_prompt = ChatPromptTemplate.from_template(
            """You are a senior software architect. Create a plan to implement: {task}.
            Available tools: read_file, write_file, run_command.

            RELEVANT CODE PATTERNS (Use these if applicable):
            {skills}

            Current Directory: {cwd}
            """
        )
        cwd = os.getcwd()
        plan_chain = planning_prompt | self.llm
        # Invoke with context
        plan = plan_chain.invoke({
            "task": task_description,
            "cwd": cwd,
            "skills": skills_context
        })

        # 2. Execute (Loop)
        # For simplicity in this v1, we let the LLM execute tools in a single turn if possible,
        # but robust implementation requires a loop (LangGraph handles this better).
        # Here we just return the plan for the Graph to execute.
        return plan.content

    def simulate_and_apply(self, code_block: str, test_cmd: str, target_file: str) -> str:
        """Advanced execution: Simulate first, then apply."""
        simulator = create_simulator()
        success, output = simulator.simulate_execution(code_block, test_cmd)

        if success:
            # Commit to real file system
            write_file(target_file, code_block)
            return f"Simulation Passed. Code applied to {target_file}.\nOutput: {output}"
        else:
            return f"Simulation Failed. Code NOT applied.\nOutput: {output}"

    def execute_code_action(self, instructions: str):
        """Directly executes tools based on instructions."""
        messages = [("system", "You are a coding agent. Use tools to execute instructions."), ("user", instructions)]
        result = self.llm_with_tools.invoke(messages)
        return result

def create_software_engineer(memory: CognitiveMemory, model_name: str = "gpt-4o") -> SoftwareEngineer:
    return SoftwareEngineer(memory, model_name)
