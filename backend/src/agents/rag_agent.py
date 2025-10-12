import asyncio

from agents import Agent, Runner, OpenAIChatCompletionsModel, set_tracing_disabled, function_tool, ModelSettings
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from openai import AsyncOpenAI
from src.tools.vector_search import search_knowledge_base
from src.core.config import get_settings
import os

set_tracing_disabled(True)
settings = get_settings()



client = AsyncOpenAI(base_url=settings.OPENAI_BASE_URL, 
                     api_key=settings.OPENAI_API_KEY)
model = OpenAIChatCompletionsModel(openai_client=client, 
                                   model="gpt-4.1")

# The agent below should answer questions related to Federal Reserve speeches.
# It is still incomplete:
# - Add specific instructions for the agent to follow when answering questions.
# - Add a function tool that performs vector search, and pass it to the agent
# - Tip: there are different function tool execution modes

@function_tool
def search_knowledge_base_tool(query: str):
    return search_knowledge_base(query)


async def main():
    agent = Agent(
        name="FedSpeechAgent",
        instructions=f"""
            {RECOMMENDED_PROMPT_PREFIX}
            You are a helpful assistant that synthesizes information from multiple sources
            to provide a comprehensive answer to the user's question.
            
            Think step by step:

            1. User will enter a question in natural language.
            2. Understand the question and generate 3 different search queries that would help find relevant information.
            3. Use the `search_knowledge_base_tool` tool to retrieve information for each of the generated queries.
            4. Synthesize the information retrieved from the `search_knowledge_base_tool` tool calls to formulate a comprehensive response.
            5. Provide the final answer along with the sources used.
            """,

        model=model,
        tools=[search_knowledge_base_tool],
        model_settings=ModelSettings(tool_choice = "required")
    )

    result = await Runner.run(agent, "What's the fed overview about monetary policy as of August 2025?")
    return result.final_output

if __name__ == "__main__":
    
    result = asyncio.run(main())
    print(result)

