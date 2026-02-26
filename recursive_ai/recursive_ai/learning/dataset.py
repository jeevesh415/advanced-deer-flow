import json
import os
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

class TrainingDataCollector:
    """Collects successful interaction traces for future model fine-tuning."""
    def __init__(self, dataset_path: str = "recursive_ai/data/finetune.jsonl"):
        self.dataset_path = dataset_path
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)

    def save_trace(self, task: str, messages: List[BaseMessage]):
        """Formats a conversation trace into ChatML/JSONL format."""

        # Convert LangChain messages to a simple list of dicts
        conversation = []
        conversation.append({"role": "system", "content": "You are a Recursive AI capable of self-improvement."})
        conversation.append({"role": "user", "content": task})

        for m in messages:
            role = "assistant" if isinstance(m, AIMessage) else "user"
            # In a graph, SystemMessages might be internal logs, but let's treat them as context
            if m.type == "system":
                 # Skip internal system logs for fine-tuning to avoid noise,
                 # or map them to "assistant" thought process if relevant.
                 # For now, we only save the final 'assistant' output or significant steps
                 continue

            content = m.content
            if content:
                conversation.append({"role": role, "content": content})

        # Append to JSONL file
        entry = {"messages": conversation}
        try:
            with open(self.dataset_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            print(f"Failed to save training data: {e}")

def create_collector() -> TrainingDataCollector:
    return TrainingDataCollector()
