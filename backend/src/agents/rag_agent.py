import asyncio
from unittest import result
import re

from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
from ..models.models import ChatRequest, ChatResponse, AgentResponse
from agents.items import ToolCallOutputItem, RunItem
import pandas as pd

from openai import AsyncOpenAI
from ..tools.vector_search import search_knowledge_base
import os

set_tracing_disabled(True)

from phoenix.otel import register
from openinference.semconv.trace import SpanAttributes
from openinference.instrumentation import OITracer
from opentelemetry.trace import StatusCode



tracer_provider = register(
  project_name="fast_api_agent",
  auto_instrument=True,
  batch=True
)

tracer: OITracer = tracer_provider.get_tracer(instrumenting_module_name = "opentelemetry.instrumentation.agents")



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

            prompt = RECOMMENDED_PROMPT_PREFIX + """
            User Query: {query}
            Answer the user's query based on the knowledge base search results.
            Provide the answer in a concise manner. Always include the sources used to generate the answer.
            """.format(query=query)


            with tracer.start_as_current_span(
                    "fed_speech_rag_agent",
                    openinference_span_kind="chain"
                ) as span:
                    try:
                        span.set_attribute(SpanAttributes.INPUT_VALUE, query)

                        agent = Agent(
                            name="FedSpeechAgent",
                            instructions=prompt,
                            model=self.model,
                            tools=[self.search_knowledge_base_tool],
                            model_settings=ModelSettings(tool_choice="search_knowledge_base_tool"),
                            output_type = AgentResponse
                        )
                        result = await Runner.run(agent, prompt)
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, str(result.final_output))
                        span.set_status(StatusCode.OK)
                        return result.final_output
                    except Exception as e:
                        span.set_attribute(SpanAttributes.OUTPUT_VALUE, f"Error: {str(e)}")
                        span.set_status(StatusCode.ERROR)
                        return AgentResponse(answer=f"Error: {str(e)}", sources=[])

        
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
