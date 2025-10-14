import asyncio
from unittest import result
import re

from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from ..schemas.requests import ChatRequest, ChatResponse
from agents.items import ToolCallOutputItem, RunItem
import pandas as pd

from openai import AsyncOpenAI
from ..tools.vector_search import search_knowledge_base
import os

set_tracing_disabled(True)

import logging
from typing import Any, List, Dict

class RAGAgent:
    """
    Retrieval-Augmented Generation (RAG) Agent for chat-based interactions.
    Encapsulates model and tools for answering user queries.
    """
    def __init__(self) -> None:
        self.client = AsyncOpenAI(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.model = OpenAIChatCompletionsModel(
            openai_client=self.client,
            model="gpt-4.1"
        )

    def _extract_sources(self, items: list[RunItem]) -> list[dict[str, Any]]:
        search_patterns = [{'column': 'Title', 'pattern':r"Title:([^\n]+)"},
                           {'column': 'Score', 'pattern':r"Score:([^\n]+)"},
                           {'column': 'Id', 'pattern':r"Id:([^\n]+)"}]
        
        sources = []
        for itam in items:
            if isinstance(itam, ToolCallOutputItem):
                tool_output = itam.output
                source_dict = {}
                for search_pattern in search_patterns:
                    pattern = search_pattern['pattern']
                    column = search_pattern['column']
                    matches = re.findall(pattern, tool_output)
                    source_dict[column] = matches
                sources_df = pd.DataFrame(source_dict)
                sources = sources_df.to_dict(orient='records')
        return sources


    async def chat(self, query: str, session_id: str = None) -> str:
        """
        Chat with the RAG agent using the provided query.

        Args:
            query (str): The user's query.
            session_id (str, optional): Session identifier for context.

        Returns:
            str: The agent's response.
        """
        try:
            agent = Agent(
                name="FedSpeechAgent",
                instructions=f"""
                {RECOMMENDED_PROMPT_PREFIX}
                You are a helpful assistant that synthesizes information from multiple sources
                to provide a comprehensive answer to the user's question.
            
                Think step by step:

                1. User will enter a question in natural language.
                2. Understand the question and generate 3 search queries that would help find relevant information.
                3. Use the `search_knowledge_base_tool` tool to retrieve information for each of the generated queries.
                4. Synthesize the information retrieved from the `search_knowledge_base_tool` tool calls to formulate a comprehensive response. Do not use a source with the same title multiple times.
                5. Provide the final answer 
                6. If you are unable to find relevant information, respond with "I don't know."

                """,
                model=self.model,
                tools=[self.search_knowledge_base_tool],
                model_settings=ModelSettings(tool_choice="required")
            )

            response = await Runner.run(agent, query)
            result = response.final_output
            sources = []
            for item in response.new_items:
                item_sources = self._extract_sources([item])
                sources.extend(item_sources)
            chat_response = {"answer": result, "sources": sources, "session_id": session_id}
            return chat_response
        
        except Exception as e:
            logging.error(f"RAGAgent chat error: {e}")
            return "Sorry, there was an error processing your request."

    @function_tool
    async def search_knowledge_base_tool(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Tool to search the knowledge base using the VectorSearchTool.

        Args:
            query (str): The search query.
            limit (int): The maximum number of results to return.

        Returns:
            list[dict]: A list of dictionaries containing document information.
        """
        return search_knowledge_base(query, limit)


def create_rag_agent() -> RAGAgent:
    """
    Factory function to create and return a RAGAgent instance.
    """
    agent = RAGAgent()

    return agent
