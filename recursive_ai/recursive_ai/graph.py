from typing import TypedDict, Annotated, List, Union
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from recursive_ai.memory.long_term import CognitiveMemory
from recursive_ai.agents.acquisition import create_research_agent
from recursive_ai.agents.evolution import create_software_engineer
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

    Decide the next step: 'research', 'code', or 'finish'.
    Return ONLY the word.
    """
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    response = llm.invoke([SystemMessage(content=prompt)])
    decision = response.content.strip().lower()

    if "research" in decision:
        return {"next_step": "research", "iterations": iterations + 1}
    elif "code" in decision:
        return {"next_step": "code", "iterations": iterations + 1}
    else:
        return {"next_step": "end"}

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

    workflow.set_entry_point("planner")

    workflow.add_conditional_edges(
        "planner",
        lambda x: x['next_step'],
        {
            "research": "researcher",
            "code": "coder",
            "finish": END,
            "end": END
        }
    )

    workflow.add_edge("researcher", "planner")
    workflow.add_edge("coder", "reviewer")

    workflow.add_conditional_edges(
        "reviewer",
        lambda x: x.get('next_step', 'planner'),
        {
            "code": "coder",
            "planner": "planner"
        }
    )

    return workflow.compile()
