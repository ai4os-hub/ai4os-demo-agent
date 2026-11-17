from starlette.applications import Starlette
from a2a.server.routes import create_agent_card_routes,create_jsonrpc_routes
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentSkill,
)
from agent_demo.core.a2a_executor import A2AExecutor


def create_agent_card(
    name: str,
    description: str,
    url: str,
    skill_id: str,
    skill_name: str,
    skill_description: str,
    tags: list[str],
    examples: list[str],
) -> AgentCard:

    # AgentCard is defined for each agent we work with
    return AgentCard(
        name=name, # same name that in litellm
        description=description,
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True), # if it is not specified, a client can assume that it should not use message/stream
        supported_interfaces=[ # tells the client where to invoke the agent
            AgentInterface(
                url=url,
                protocol_binding="JSONRPC",      
                protocol_version="0.3", # if litellm is using A2A 0.3:
            )
        ],
        skills=[
            AgentSkill( # describes an agent capability, used when discovering
                id=skill_id,
                name=skill_name,
                description=skill_description,
                tags=tags,
                examples=examples,
            )
        ],
    )




def create_a2a_app(
    *,
    backend,
    agent_card: AgentCard
):
    # create an executor object of the general class
    executor = A2AExecutor(backend=backend)

    # DefaultRequestHandler: this is the standard a2a handler, receives the json-rpc request and decides how to transform it into an agent execution
    # InMemoryTaskStore: saves tasks in memory 
    handler = DefaultRequestHandler(
        agent_executor=executor, # what code execute when a request is received
        task_store=InMemoryTaskStore(), # where to save tasks
        agent_card=agent_card, # which agent is publishing
    )

    # final a2a app is constructed with this
    app = Starlette(
        routes=[ # two routing groups
            *create_agent_card_routes(agent_card), # exposes the card, mounted in /XXXX
            *create_jsonrpc_routes( # exposes the a2a endpoint, mounted in /XXXX
                handler,
                "/",
                enable_v0_3_compat=True, # client uses message/stream and /send, and tasks/get, that are from previous versions
            ),
        ]
    )

    return app