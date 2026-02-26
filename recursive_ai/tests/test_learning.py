import unittest
import os
import json
from unittest.mock import MagicMock
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from recursive_ai.learning.dataset import TrainingDataCollector

class TestDataset(unittest.TestCase):
    def test_save_trace(self):
        """Test that traces are saved to JSONL."""
        test_path = "recursive_ai/data/test_dataset.jsonl"
        collector = TrainingDataCollector(dataset_path=test_path)

        # In dataset.py:
        # conversation.append({"role": "system", ...}) -> idx 0
        # conversation.append({"role": "user", content: task}) -> idx 1

        # Messages loop:
        # HumanMessage("Task A") -> role="user" -> idx 2
        # AIMessage("Plan A") -> role="assistant" -> idx 3
        # SystemMessage -> skipped

        messages = [
            HumanMessage(content="Task A"),
            AIMessage(content="Plan A"),
            SystemMessage(content="Debug Info") # Should be skipped
        ]

        collector.save_trace("Task A", messages)

        self.assertTrue(os.path.exists(test_path))

        with open(test_path, "r") as f:
            line = f.readline()
            data = json.loads(line)

            self.assertEqual(data["messages"][1]["role"], "user")
            self.assertEqual(data["messages"][1]["content"], "Task A")

            # idx 2 is the HumanMessage "Task A" (redundant but correct logic)
            self.assertEqual(data["messages"][2]["role"], "user")
            self.assertEqual(data["messages"][2]["content"], "Task A")

            # idx 3 is the AIMessage
            self.assertEqual(data["messages"][3]["role"], "assistant")
            self.assertEqual(data["messages"][3]["content"], "Plan A")

            # idx 4 should not exist
            self.assertTrue(len(data["messages"]) == 4)

        # Cleanup
        os.remove(test_path)

if __name__ == "__main__":
    unittest.main()
