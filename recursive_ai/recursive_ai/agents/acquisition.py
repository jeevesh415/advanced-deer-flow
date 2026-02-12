from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
import requests
from bs4 import BeautifulSoup
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

    def visit_page(self, url: str) -> str:
        """Deep scrapes a webpage for content."""
        try:
            headers = {"User-Agent": "RecursiveAI-Researcher/1.0"}
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code != 200:
                return f"Error: Status {response.status_code}"

            soup = BeautifulSoup(response.text, 'html.parser')
            # Remove scripts and styles
            for script in soup(["script", "style"]):
                script.decompose()

            text = soup.get_text()
            # Clean whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            return text[:10000] # Limit context window usage
        except Exception as e:
            return f"Error visiting {url}: {e}"

    def perform_research(self, query: str, depth: str = "brief") -> str:
        """Conducts research on a topic."""
        self.status = AgentStatus.WORKING

        # 1. Search
        try:
            results = self.tavily.invoke({"query": query})
        except Exception:
            # Fallback
            results = self.ddg.invoke(query)

        # 2. Deep Dive (if depth="deep")
        context_content = str(results)
        if depth == "deep" and isinstance(results, list):
            # Try to visit the first valid link
            try:
                # Basic heuristic to find a url in the result dict
                url = next((res.get('url') or res.get('link') for res in results if res.get('url') or res.get('link')), None)
                if url:
                    page_content = self.visit_page(url)
                    context_content += f"\n\n--- Deep Dive into {url} ---\n{page_content}"
            except Exception:
                pass

        # 3. Synthesize
        synthesis_prompt = ChatPromptTemplate.from_template(
            """You are an elite research analyst. Synthesize the following search results into a detailed report on '{query}'.
            Focus on technical details, architectural patterns, and actionable code snippets.

            Results: {results}

            Report:"""
        )
        chain = synthesis_prompt | self.llm | StrOutputParser()
        report = chain.invoke({"query": query, "results": context_content})

        # 4. Store in Memory
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
