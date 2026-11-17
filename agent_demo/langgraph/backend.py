from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from agent_demo.core.events import AgentStreamEvent
from agent_demo.langgraph.conversation_formatting import convert_frontend_history

# to adapt LangGraph to A2AExecutor
class LangGraphBackend:

    def __init__(self, agent):
        self.agent = agent # it is the compiled LangGraph graph

    async def stream(self, query, *, message_history, context_id):
        # to adapt frontend format to HumanMessage/AIMessage
        messages = convert_frontend_history(message_history)
        # adds the current message
        messages.append(HumanMessage(content=query))
        final_state = None

        # to execute LangGraph in streaming
        async for mode, data in self.agent.astream({"messages": messages}, stream_mode=["messages", "values"]):
            if mode == "messages":
                chunk, metadata = data # metadata has info about where it was produced, the node, execution... (not used by now)
                if not isinstance(chunk, AIMessageChunk): # it filters chunks from LLM to make them visible, and not information regarding tools (ToolMessage), HumanMessage, AIMessage...
                    continue
                reasoning = chunk.additional_kwargs.get("reasoning_content") # it extracts thinking
                if reasoning:
                    yield AgentStreamEvent("thinking", reasoning)
                if isinstance(chunk.content, str): # it extracts normal responses
                    if chunk.content:
                        yield AgentStreamEvent("response", chunk.content)
                elif isinstance(chunk.content, list): # responses can also have a list format
                    for block in chunk.content:
                        if (isinstance(block, dict) and block.get("type") == "text"):
                            text = block.get("text", "")
                            if text:
                                yield AgentStreamEvent("response", text)
            elif mode == "values":
                final_state = data # data is the state of the graph at that moment

        # to recover the complete record
        if final_state is not None:
            updated_history = final_state["messages"]
            output = ""
            for message in reversed(updated_history): # reversed to find the last message from the assistant
                if isinstance(message, AIMessage):
                    output = message.text
                    break
            yield AgentStreamEvent("final", output, updated_history)