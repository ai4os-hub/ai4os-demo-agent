import httpx
import os
from pathlib import Path
from dotenv import load_dotenv
from dataclasses import dataclass

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)
LITELLM_BASE_URL = os.environ["LITELLM_BASE_URL"]


@dataclass(frozen=True)
class MCPConnection:
    name: str
    url: str
    headers: dict[str, str]


def get_mcp_connection(mcp_name: str, api_key: str) -> MCPConnection:
    return MCPConnection(
        name=mcp_name,
        url=f"{LITELLM_BASE_URL}/{mcp_name}/mcp",
        headers={"x-litellm-api-key": f"Bearer {api_key}"},
    )