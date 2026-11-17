import os
import logging
from contextlib import asynccontextmanager, AsyncExitStack
from fastapi import FastAPI
from dotenv import load_dotenv
from pathlib import Path
import inspect
from agent_demo.core.a2a_app_creator import create_agent_card, create_a2a_app
from a2a.server.request_handlers.response_helpers import agent_card_to_dict
from agent_demo.core.a2a_executor import A2AExecutor
from agent_demo.pydantic_ai.backend import PydanticAIBackend
from agent_demo.pydantic_ai.agent import create_pydanticai_agent
from agent_demo.langgraph.backend import LangGraphBackend
from agent_demo.langgraph.agent import create_langgraph_agent
from fastapi.responses import FileResponse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

LITELLM_BASE_URL = os.environ["LITELLM_BASE_URL"]
LITELLM_API_KEY = os.environ["LITELLM_API_KEY"]
LOCAL_BASE_URL = os.environ["LOCAL_BASE_URL"]
LITELLM_AGENT_MODEL = os.environ['LITELLM_AGENT_MODEL']
logger = logging.getLogger("uvicorn.error")

pydanticai_a2a_agent_card = create_agent_card(
    name="pydanticai-demo",
    description="A demo intelligent guide to the AI4EOSC ecosystem. It navigates the available documentation to help the AI4EOSC community understand how the platform works, providing answers grounded in official AI4EOSC resources.",
    url=f"{LOCAL_BASE_URL}/pydanticai-demo/",
    skill_id="documentation-info",
    skill_name="Documentation assistant",
    skill_description="Answers AI4EOSC platform-related questions.",
    tags=["documentation", "search", "streaming"],
    examples=[
        "How can I register in the AI4EOSC platform?",
        "How can I develop a new AI module?",
        "How can I leverage the AI4EOSC LLM in my own services?",
    ],
)

langgraph_a2a_agent_card = create_agent_card(
    name="langgraph-demo",
    description="A demo intelligent guide to the AI4EOSC ecosystem. It navigates the available documentation to help the AI4EOSC community understand how the platform works, providing answers grounded in official AI4EOSC resources.",
    url=f"{LOCAL_BASE_URL}/langgraph-demo/",
    skill_id="agent_langgraph_demo",
    skill_name="Agent langgraph demo",
    skill_description="Answers AI4EOSC platform-related questions.",
    tags=["documentation", "search", "streaming"],
    examples=[
        "How can I register in the AI4EOSC platform?",
        "How can I develop a new AI module?",
        "How can I leverage the AI4EOSC LLM in my own services?",
    ],
)

agent_configs = {
    "pydanticai-demo": {
        "card": pydanticai_a2a_agent_card,
        "factory": create_pydanticai_agent,
        "backend_factory": PydanticAIBackend,
        'model': LITELLM_AGENT_MODEL
    },
    "langgraph-demo": {
        "card": langgraph_a2a_agent_card,
        "factory": create_langgraph_agent,
        "backend_factory": LangGraphBackend,
        'model': LITELLM_AGENT_MODEL
    }
}

async def create_local_backends():

    backends = {}
    logger.info("Creating %d local agents", len(agent_configs))

    for agent_name, config in agent_configs.items():
        logger.info("Creating local agent: %s", agent_name)

        # create the agent using its framework
        framework_agent = config["factory"](config["model"], LITELLM_API_KEY) 

        # this is needed for the langgraph agent, which is created asynchronously
        if inspect.isawaitable(framework_agent):
            framework_agent = await framework_agent

        # adapt the agent to the common A2A backend interface
        backend = config["backend_factory"](framework_agent)
        backends[agent_name] = backend

        logger.info("Local agent %s created successfully", agent_name)

    return backends

@asynccontextmanager
async def lifespan(app: FastAPI):

    logger.info("Starting local A2A agents...")
    # agents are created with a common interface
    backends = await create_local_backends()

    # agents are mounted at their respective endpoints
    for agent_name, config in agent_configs.items():
        a2a_app = create_a2a_app(backend=backends[agent_name], agent_card=config["card"])
        app.mount(f"/{agent_name}", a2a_app)
        logger.info("Mounted local agent %s at /%s", agent_name, agent_name)

    logger.info("All local agents initialized")
    yield
    logger.info("Shutting down local A2A server")


# This means that when the app starts up, FastAPI runs the lifespan, and when it closes, it cleans up using that lifespan
app = FastAPI(title="A2A Demo Server", lifespan = lifespan)

# The health endpoint is created to check whether the server is up
@app.get("/health")
def health():
    return {
        "status": "ok",
        "agents": [agent_name for agent_name, _ in agent_configs.items()]
    }
