import os
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime
from pydantic_ai import Agent
from pydantic_ai.mcp import MCPToolset
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider 
from pydantic_ai.capabilities import Thinking
from pydantic_ai.profiles.openai import OpenAIModelProfile
from agent_demo.core.mcp_call import get_mcp_connection#call_mcp_tool

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

LITELLM_BASE_URL = os.environ["LITELLM_BASE_URL"]


# to ensure there are no inconsistencies with dates due to the model you are using
def get_current_datetime() -> str:
    return datetime.now().astimezone().isoformat()


def create_pydanticai_agent(agent_model, api_key):

    # i use one of the models available on the ai4eosc platform
    model = OpenAIChatModel(
        agent_model, 
        provider=OpenAIProvider(
            base_url = LITELLM_BASE_URL, 
            api_key = api_key
        ),
        profile = OpenAIModelProfile(supports_thinking = True)
    )

    knowledge_milvus_mcp = get_mcp_connection("knowledge_milvus", api_key)

    knowledge_milvus_toolset = MCPToolset(
        knowledge_milvus_mcp.url,
        headers=knowledge_milvus_mcp.headers,
        include_instructions=True,
    )

    # I create the agent using the LLM, the instructions and the tools
    return Agent(
        model=model,
        capabilities = [Thinking(effort='high')],
        instructions=(
            f"Current date and time: {get_current_datetime()} "
            "You are an assistant specialized in the AI4EOSC platform. "

            # to establish plan mode
            "Before taking action or using any tools/assets, first assess whether the user's request is sufficiently specified. "
            "If important information is missing, ambiguous, or could materially affect the result, ask concise and targeted clarifying questions before proceeding. "
            "Do not ask for clarification when the missing information is minor or can be reasonably inferred. In those cases, state any relevant assumptions and proceed. "
            "Do not ask the user for information they have already provided earlier in the conversation. "
            "For complex or multi-step tasks, first formulate a brief internal plan identifying the goal, constraints, missing information, and required actions. Only ask the user questions when their input is necessary to resolve a meaningful ambiguity. "

            "You MUST use the available knowledge_milvus MCP tools whenever the user asks "
            "about the AI4EOSC project, its services, documentation, platform features, "
            "technical procedures, configuration, usage, deployment, APIs, tools, or how "
            "to perform any task within AI4EOSC. The collection that is usually used with "
            "the search tool is ai4eosc_docs, which has information about the platform and its use, "
            "unless the user asks for specific information. "

            "For factual questions about AI4EOSC, do not rely on internal model knowledge "
            "when the information can be retrieved from the documentation. "
            "With the search tool you can perform vector RAG searches across one or more Milvus collections. "

            "Base your factual answers about AI4EOSC on the information returned by the MCP tools. "
            "Do not invent, infer, or assume undocumented AI4EOSC-specific information. "

            "If the retrieved documentation does not contain enough information to answer "
            "the question reliably, clearly state that the available documentation does not "
            "provide a sufficient answer. "

            "After receiving tool results, synthesize them into a clear final answer rather "
            "than simply reproducing the raw retrieved content. "
            "Please, at the end of each paragraph, cite the sources used to elaborate that paragraph (if any). Use standard markdown footnote syntax to cite the sources."

            "Answer in the same language as the user. "
            "Keep the final answer concise unless the user asks for more detail. "

            "When possible, use the same language as the user's prompt for visible reasoning content."
        ),
        toolsets = [knowledge_milvus_toolset],
        output_type=str
    )