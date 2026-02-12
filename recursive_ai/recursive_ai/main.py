import sys
import argparse
from langchain_core.messages import HumanMessage
from recursive_ai.graph import create_graph

def main():
    parser = argparse.ArgumentParser(description="Recursive AI Autonomous System")
    parser.add_argument("task", type=str, help="The high-level goal for the AI.")
    parser.add_argument("--iterations", type=int, default=10, help="Max iterations.")
    args = parser.parse_args()

    print(f"🚀 Initializing Recursive AI with task: {args.task}")

    workflow = create_graph()

    initial_state = {
        "messages": [HumanMessage(content=args.task)],
        "task": args.task,
        "iterations": 0,
        "next_step": "start"
    }

    print("🧠 Cortex active. Thinking...")
    try:
        for event in workflow.stream(initial_state):
            for key, value in event.items():
                print(f"\n--- Node: {key} ---")
                if "messages" in value:
                    print(f"Output: {value['messages'][-1].content[:200]}...") # Truncate for readability
                if "next_step" in value:
                    print(f"Decision: {value['next_step']}")
    except Exception as e:
        print(f"❌ Critical Error: {e}")

if __name__ == "__main__":
    main()
