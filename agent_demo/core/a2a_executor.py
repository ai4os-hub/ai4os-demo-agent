import json
import asyncio 
import logging
from a2a.server.agent_execution.agent_executor import AgentExecutor
from a2a.server.agent_execution.context import RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from google.protobuf.json_format import MessageToDict
# the following simplify a2a objects creation
from a2a.helpers import (
    get_message_text, # extracts user text
    new_task_from_user_message, # creates a new task from the user message
    new_text_message, # creates a text a2a message
    new_text_part, # creates a textual part for an artifact 
)
from dataclasses import dataclass

logger = logging.getLogger("uvicorn.error")


# AgentExecutor is the base class that must be implemented
# RequestContext contains the request information (user message, current task, IDs, etc.)
# EventQueue is the queue to which the agent sends events (in streaming, events are queued, and the SDK converts them to SSE)
class A2AExecutor(AgentExecutor):

    def __init__(self, backend):
        self.backend = backend

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:

        # I add two channels: one for the thought itself and another for the response itself
        channels = {
            "thinking": {"created": False, "pending": None, "full_text": ""},
            "response": {"created": False, "pending": None, "full_text": ""}
        }

        logger.info("CONTEXT ID: %r", context.message.context_id)
        logger.info("CONTEXT METADATA: %r", context.metadata)
        logger.info("MESSAGE METADATA: %r", context.message.metadata)
        logger.info("WHOLE MESSAGE: %r", context.message)
        
        # 1. First, there must be a Task in the EventQueue.
        if context.current_task:
            task = context.current_task
        else:
            task = new_task_from_user_message(context.message)
            await event_queue.enqueue_event(task)

        task_id = task.id
        context_id = task.context_id

        # emits a2a events associated with that task without having to create them manually
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=task_id,
            context_id=context_id,
        )

        query = get_message_text(context.message) # extracts the user’s text (it is converted into a normal string which is passed to pydantic ai)
        logger.info("A2A agent query=%r", query) 

        # this is where the Pydantic AI part begins (this block acts as an adapter between the Pydantic AI stream and the a2a stream)
        try:

            # 2. Status updates and artefact updates can now be issued.
            await updater.start_work( # with this, the client receives something like data: {"result": {"kind": "status-update", "status": {"state": "working"}}}
                message=new_text_message( # becomes a message from the agent associated with that task
                    "Processing request.",
                    context_id = context_id,
                    task_id = task_id,
                ) 
            )

            metadata = MessageToDict(context.message.metadata, preserving_proto_field_name=True) if context.message.metadata else {}
            raw_history = metadata.get("conversation_history",[])
            logger.info("Context %s: previous messages=%d", context_id, len(raw_history))

            run_result = None

            # to emit an a2a chunk
            async def send_channel_chunk(channel: str, text: str, *, last_chunk: bool) -> None:
                state = channels[channel]
                append = state["created"]
                logger.warning("A2A SEND channel=%s append=%s last_chunk=%s text=%r", channel, append, last_chunk, text)
                await updater.add_artifact(
                    parts=[new_text_part(text, media_type="text/plain")],
                    artifact_id=channel,
                    name=channel,
                    metadata={"channel": channel},
                    append=append,
                    last_chunk=last_chunk,
                )
                state["created"] = True

            # function to retain the last chunk
            async def push_channel_chunk(channel: str, text: str | None) -> None:
                if not text:
                    return
                state = channels[channel]
                if state["pending"] is not None:
                    await send_channel_chunk(channel, state["pending"], last_chunk=False)
                state["pending"] = text
                state["full_text"] += text

            # function to close the channel
            async def finish_channel(channel: str) -> None:
                state = channels[channel]
                if state["pending"] is None:
                    return
                await send_channel_chunk(
                    channel,
                    state["pending"],
                    last_chunk=True,
                )
                state["pending"] = None
            
            # I define heartbeats to avoid frontend to close the connection
            async def heartbeat():
                counter = 0

                try:
                    while True:
                        await asyncio.sleep(10) # emits a heartbeat each 10 seconds
                        counter += 1
                        logger.warning("A2A HEARTBEAT %d task=%s", counter, task_id)
                        await updater.start_work(
                            message=new_text_message(
                                f"Still working... heartbeat {counter}",
                                context_id=context_id,
                                task_id=task_id,
                            )
                        )

                except asyncio.CancelledError:
                    logger.warning("A2A HEARTBEAT STOPPED task=%s", task_id)
                    raise

            updated_history = None
            heartbeat_task = asyncio.create_task(heartbeat())

            try:
                # The stream is opened, already converted to a common format
                async for event in self.backend.stream(query, message_history=raw_history, context_id=context_id):

                        if event.type == "thinking":
                            await push_channel_chunk("thinking", event.content)
                        elif event.type == "response":
                            # by the time it starts writing the final answer, the thinking process is already over
                            await finish_channel("thinking")
                            await push_channel_chunk("response", event.content)

                        elif event.type == "final":
                            output = event.content
                            updated_history = event.history

                            # Fallback only if no incremental response was received
                            if output and not channels["response"]["full_text"]:
                                await push_channel_chunk("response", output)

            finally:
                heartbeat_task.cancel()

                try:
                    await heartbeat_task
                except asyncio.CancelledError:
                    pass

            await finish_channel("thinking")
            await finish_channel("response")
                
            # to retrieve info about the number of saved messages
            if updated_history is not None:
                logger.info("Context %s: saved messages=%d", context_id, len(updated_history))

            logger.warning("FINAL THINKING=%r", channels["thinking"]["full_text"])
            logger.warning("FINAL RESPONSE=%r", channels["response"]["full_text"])

            # the task is marked as completed
            await updater.complete(
                message=new_text_message(
                    "Completed.",
                    context_id=context_id,
                    task_id=task_id,
                )
            )

        # if the agent fails, a ‘failed’ status is returned
        except Exception as e:
            logger.exception("A2A agent failed")
            await updater.failed(
                message=new_text_message(
                    f"{type(e).__name__}: {e}",
                    context_id=context_id,
                    task_id=task_id,
                )
            )

    # this is required by AgentExecutor, and is executed if the a2a client requests the cancellation of a task (I am not currently using it)
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id,
            context_id=context.context_id,
        )
        await updater.cancel()
