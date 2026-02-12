# Recursive AI (RAI)

**Recursive AI** is a next-generation autonomous artificial intelligence system. Unlike traditional agents that perform tasks and wait for commands, RAI is designed to be **self-improving, resource-aware, and continuously evolving**.

## Core Philosophy

1.  **Autonomous Evolution**: RAI actively seeks out new knowledge (Arxiv, web) and integrates it into its own codebase.
2.  **Meta-Cognition**: A high-level "Cortex" manages resources, time, and specialized agent groups.
3.  **Strict Typing**: All internal communication is strictly typed and validated, preventing hallucinations and errors.
4.  **Self-Correction**: RAI writes tests for its own code changes and rolls back failures automatically.

## Architecture

*   **Cortex (Orchestrator)**: The central brain managing goals and resources.
*   **Acquisition Group**: Research agents specialized in finding and synthesizing new knowledge.
*   **Evolution Group**: Engineering agents that read, write, and test the system's own source code.
*   **Memory (RAG)**: Long-term storage for learned concepts using ChromaDB.

## Usage

1.  Set up environment: `uv sync` or `pip install .`
2.  Set `TAVILY_API_KEY` and `OPENAI_API_KEY` (or compatible LLM keys).
3.  Run the main loop: `python -m recursive_ai.main`

## Disclaimer

This is an experimental system capable of modifying its own code. Run in a sandboxed environment.
