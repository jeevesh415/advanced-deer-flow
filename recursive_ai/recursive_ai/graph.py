from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from recursive_ai.memory.long_term import CognitiveMemory
from recursive_ai.agents.acquisition import create_research_agent
from recursive_ai.agents.evolution import create_software_engineer
from recursive_ai.agents.scientist import create_scientist
from recursive_ai.agents.reflector import create_reflector
from recursive_ai.learning.dataset import create_collector
import operator

# Define State
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    next_step: str
    iterations: int
    task: str

# Define Nodes
def planner_node(state: AgentState):
    """Decides the next action based on state."""
    messages = state['messages']
    task = state.get('task', "No task")

    # Simple logic for now: alternating research and coding
    # In a real advanced system, use an LLM to decide dynamically
    current_step = state.get('next_step', 'start')
    iterations = state.get('iterations', 0)

    if iterations > 5:
        return {"next_step": "end"}

    prompt = f"""You are the Cortex of an autonomous AI.
    Current Task: {task}
    History: {messages[-2:] if len(messages)>1 else messages}

    Decide the next step:
    - 'research': To gather information.
    - 'code': To implement a solution.
    - 'experiment': To test abstract hypotheses in the lab (use this for optimization or unknowns).
    - 'finish': If the task is complete.

    Return ONLY the word.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke([SystemMessage(content=prompt)])
    decision = response.content.strip().lower()

    if "research" in decision:
        return {"next_step": "research", "iterations": iterations + 1}
    elif "code" in decision:
        return {"next_step": "code", "iterations": iterations + 1}
    elif "experiment" in decision:
        return {"next_step": "experiment", "iterations": iterations + 1}
    else:
        return {"next_step": "end"}

def experiment_node(state: AgentState):
    """Executes abstract experiments."""
    task = state['task']
    try:
        memory = CognitiveMemory()
        scientist = create_scientist(memory)
        report = scientist.conduct_experiment(task)
        return {"messages": [SystemMessage(content=f"Experiment Report: {report}")]}
    except Exception as e:
        return {"messages": [SystemMessage(content=f"Experiment Failed: {e}")]}

def reflector_node(state: AgentState):
    """Analyzes the run, updates strategy, and saves training data."""
    task = state['task']
    messages = state['messages']
    reflector = create_reflector()

    # 1. Update Strategy
    insight = reflector.reflect_on_execution(messages, task, success=True) # Assume success if we reached here

    # 2. Save Training Data
    try:
        collector = create_collector()
        collector.save_trace(task, messages)
        data_msg = "Run saved to finetuning dataset."
    except Exception as e:
        data_msg = f"Failed to save dataset: {e}"

    return {"messages": [SystemMessage(content=f"Meta-Cognition Insight: {insight}. {data_msg}")]}

def review_node(state: AgentState):
    """Reviews code and runs tests."""
    task = state['task']
    try:
        memory = CognitiveMemory()
        agent = create_software_engineer(memory)

        # In a real scenario, we would look for test files related to the task
        # For now, we ask the agent to verify its own work
        verification = agent.execute_code_action(f"Verify the implementation for task: {task}. Run tests if available.")

        if "Error" in str(verification) or "Fail" in str(verification):
            return {"next_step": "code", "messages": [SystemMessage(content=f"Review Failed: {verification}")]}
        else:
            return {"next_step": "planner", "messages": [SystemMessage(content=f"Review Passed: {verification}")]}

    except Exception as e:
        return {"next_step": "planner", "messages": [SystemMessage(content=f"Review Error: {e}")]}

def research_node(state: AgentState):
    """Executes research."""
    task = state['task']
    # Use Memory (should be configured via env vars)
    try:
        memory = CognitiveMemory()
        agent = create_research_agent(memory)
        result = agent.perform_research(task)
    except Exception as e:
        result = f"Research Error: {e}"

    return {"messages": [SystemMessage(content=f"Research Result: {result}")]}

def code_node(state: AgentState):
    """Executes coding."""
    task = state['task']
    try:
        memory = CognitiveMemory()
        agent = create_software_engineer(memory)

        # We ask the agent to implement based on the last research
        if state['messages']:
            last_message = state['messages'][-1].content
        else:
            last_message = "No prior context."

        plan = agent.implement_feature(f"Task: {task}. Context: {last_message}")

        # Execute the plan (simplified for now, ideally strictly parsed)
        result = agent.execute_code_action(plan)
    except Exception as e:
        result = f"Coding Error: {e}"

    return {"messages": [SystemMessage(content=f"Code Execution Result: {result}")]}

# Build Graph
def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("coder", code_node)
    workflow.add_node("reviewer", review_node)
    workflow.add_node("experimenter", experiment_node)
    workflow.add_node("reflector", reflector_node)

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        lambda x: x['next_step'],
        {
            "research": "researcher",
            "code": "coder",
            "experiment": "experimenter",
            "finish": "reflector",
            "end": "reflector"
        }
    )

    workflow.add_edge("researcher", "planner")
    workflow.add_edge("coder", "reviewer")
    workflow.add_edge("experimenter", "planner")

    workflow.add_conditional_edges(
        "reviewer",
        lambda x: x.get('next_step', 'planner'),
        {
            "code": "coder",
            "planner": "planner"
        }
    )

    workflow.add_edge("reflector", END)

    return workflow.compile()
