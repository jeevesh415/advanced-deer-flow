from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from recursive_ai.core.protocol import Message, Task, Observation, AgentStatus
from recursive_ai.memory.long_term import CognitiveMemory

class ResearchAgent:
    """Agent specialized in finding and synthesizing information."""
    def __init__(self, memory: CognitiveMemory, model_name: str = "gpt-4o"):
        self.memory = memory
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tavily = TavilySearchResults(max_results=5)
        self.ddg = DuckDuckGoSearchRun()
        self.status = AgentStatus.IDLE

    def perform_research(self, query: str, depth: str = "brief") -> str:
        """Conducts research on a topic."""
        self.status = AgentStatus.WORKING

        # 1. Search
        try:
            results = self.tavily.invoke({"query": query})
        except Exception:
            # Fallback
            results = self.ddg.invoke(query)

        # 2. Synthesize
        synthesis_prompt = ChatPromptTemplate.from_template(
            """You are an elite research analyst. Synthesize the following search results into a detailed report on '{query}'.
            Focus on technical details, architectural patterns, and actionable code snippets.

            Results: {results}

            Report:"""
        )
        chain = synthesis_prompt | self.llm | StrOutputParser()
        report = chain.invoke({"query": query, "results": str(results)})

        # 3. Store in Memory
        self.memory.add_memory(
            text=report,
            metadata={"source": "research_agent", "query": query, "type": "report"}
        )

        self.status = AgentStatus.IDLE
        return report

    def find_latest_papers(self, topic: str) -> List[Dict]:
        """Specific search for Arxiv papers (simulated via search engine)."""
        query = f"site:arxiv.org {topic} latest papers 2024 2025"
        try:
            results = self.tavily.invoke({"query": query})
            return results
        except Exception:
            return [{"error": "Search failed"}]

def create_research_agent(memory: CognitiveMemory, model_name: str = "gpt-4o") -> ResearchAgent:
    return ResearchAgent(memory, model_name)
