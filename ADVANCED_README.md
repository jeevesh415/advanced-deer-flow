# advanced-deer-flow: Hierarchical Research Agent with Knowledge Graph

This repository is an advanced fork of the original `deer-flow`, upgraded to implement a state-of-the-art **Hierarchical Multi-Agent System** for deep research and knowledge synthesis.

## Key Advanced Features

1.  **Hierarchical Agent Architecture**: The system operates with a **Manager Agent** that receives the research query, breaks it down into sub-tasks, and delegates them to specialized **Worker Agents** (e.g., Search Agent, Summarization Agent, Synthesis Agent). This ensures more focused and efficient research.
2.  **Knowledge Graph Integration**: Instead of simply summarizing findings, the Synthesis Agent now constructs a **Knowledge Graph (KG)** from the collected data. This allows for complex querying, relationship discovery, and a more structured, verifiable output.
3.  **Model Context Protocol (MCP) Ready**: The agent communication layer is designed to be easily integrated with the Model Context Protocol (MCP), allowing this research flow to be a specialized tool within a larger agent ecosystem.

## Initial Structural Changes

This initial commit includes the original codebase and a placeholder for the new hierarchical and KG modules.

*   **`src/manager_agent/`**: Placeholder for the top-level orchestration logic.
*   **`src/worker_agents/`**: Placeholder for specialized worker agent definitions.
*   **`src/knowledge_graph/`**: Placeholder for KG construction and storage logic (e.g., using a graph database client).

## Next Steps

The next phase of development will focus on implementing the Manager Agent's task decomposition logic and integrating a lightweight graph database for the Knowledge Graph.
