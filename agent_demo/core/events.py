from dataclasses import dataclass
from typing import Literal, Any

@dataclass
class AgentStreamEvent:
    type: Literal["thinking", "response", "final"]
    content: str = ""
    history: Any = None

@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str
