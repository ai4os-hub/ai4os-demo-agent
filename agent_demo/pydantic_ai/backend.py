# these are the events produced by agent.run_stream_events(query)
from agent_demo.core.events import AgentStreamEvent
from pydantic_ai.run import AgentRunResultEvent
from pydantic_ai import Agent
from pydantic_ai.messages import PartStartEvent, PartDeltaEvent, ThinkingPart, ThinkingPartDelta, TextPart, TextPartDelta
from agent_demo.pydantic_ai.conversation_formatting import convert_frontend_history

class PydanticAIBackend:

    def __init__(self, agent: Agent):
        self.agent = agent

    async def stream(self, query, *, message_history, context_id):

        async with self.agent.run_stream_events(query, message_history=convert_frontend_history(message_history), conversation_id=context_id) as events:
            async for event in events:

                if isinstance(event, PartStartEvent):
                    if isinstance(event.part, ThinkingPart):
                        yield AgentStreamEvent("thinking", event.part.content)
                    elif isinstance(event.part, TextPart):
                        # by the time it starts writing the final answer, the thinking process is already over
                        yield AgentStreamEvent("response", event.part.content)

                elif isinstance(event, PartDeltaEvent):
                    if isinstance(event.delta, ThinkingPartDelta):
                        yield AgentStreamEvent("thinking", event.delta.content_delta)
                    elif isinstance(event.delta, TextPartDelta):
                        yield AgentStreamEvent("response", event.delta.content_delta)

                elif isinstance(event, AgentRunResultEvent):
                    yield AgentStreamEvent("final", str(event.result.output), event.result.all_messages())