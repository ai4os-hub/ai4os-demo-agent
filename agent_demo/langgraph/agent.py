import os
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime
from langchain_litellm import ChatLiteLLM
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain.mcp import MCPAdapter
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from agent_demo.core.mcp_call import get_mcp_connection
from langchain_core.runnables import RunnableConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)
LITELLM_BASE_URL = os.environ["LITELLM_BASE_URL"]


# to ensure there are no inconsistencies with dates due to the model you are using
def get_current_datetime() -> str:
    return datetime.now().astimezone().isoformat()


async def create_langgraph_agent(agent_model, api_key):

    model = ChatLiteLLM(
        model=f"litellm_proxy/{agent_model}", # to not directly call a supplier
        api_base=LITELLM_BASE_URL,
        api_key=api_key,
        reasoning_effort="high",
    )

    knowledge_milvus_mcp = get_mcp_connection("knowledge_milvus", api_key)

    mcp_config = {
        "mcpServers": {
            knowledge_milvus_mcp.name: {
                "url": knowledge_milvus_mcp.url,
                "headers": knowledge_milvus_mcp.headers,
            }
        }
    }

    async with MCPAdapter(mcp_config) as adapter:
        tools = await adapter.list_tools()

    
    # the model now has access to the tool, but it is not executed through bind_tools
    model_with_tools = model.bind_tools(tools) 

    content = (
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
    )

    # it is a node from the graph
    # RunnableConfig is particularly important for streaming
    async def call_model(state: MessagesState, config: RunnableConfig):
        messages = [SystemMessage(content=content), *state["messages"]] # creates the messages that are sent to the model
        response = await model_with_tools.ainvoke(messages, config=config) # the model is called
        return {"messages": [response]}

    graph = StateGraph(MessagesState) # it creates the graph builder
    graph.add_node("model", call_model) # when LangGraph reaches "model" node, it executes call_model
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "model")
    graph.add_conditional_edges("model", tools_condition, {"tools": "tools", "__end__": END}) # it decides whether to call a tool or not
    graph.add_edge("tools", "model") # after calling a tool, the model has to be called to produce the response

    return graph.compile()
