import unittest
import os
from unittest.mock import MagicMock, patch
from recursive_ai.graph import create_graph
from recursive_ai.agents.acquisition import ResearchAgent
from recursive_ai.agents.evolution import SoftwareEngineer
from recursive_ai.memory.long_term import CognitiveMemory

class TestRecursiveAI(unittest.TestCase):
    def setUp(self):
        # Set dummy keys to prevent instantiation errors
        os.environ["OPENAI_API_KEY"] = "sk-dummy-key"
        os.environ["TAVILY_API_KEY"] = "tv-dummy-key"

    def test_graph_compilation(self):
        """Test that the graph compiles without error."""
        try:
            graph = create_graph()
            self.assertIsNotNone(graph)
        except Exception as e:
            self.fail(f"Graph compilation failed: {e}")

    @patch("recursive_ai.agents.acquisition.DuckDuckGoSearchRun")
    @patch("recursive_ai.agents.acquisition.TavilySearchResults")
    def test_agent_instantiation(self, MockTavily, MockDDG):
        """Test that agents can be created."""
        memory = MagicMock(spec=CognitiveMemory)

        researcher = ResearchAgent(memory)
        self.assertIsNotNone(researcher)

        engineer = SoftwareEngineer(memory)
        self.assertIsNotNone(engineer)

if __name__ == "__main__":
    unittest.main()
